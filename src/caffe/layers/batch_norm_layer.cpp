#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);    // channels_ is C
  eps_ = param.eps();
  if (this->blobs_.size() > 0)
  {
    LOG(INFO) << "Skipping parameter initialization";
  }
  else
  {
    this->blobs_.resize(3);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));     // store mua(k), shape: (C,)
    this->blobs_[1].reset(new Blob<Dtype>(sz));     // store sigmaa^2(k), shape: (C,)
    sz[0] = 1;
    this->blobs_[2].reset(new Blob<Dtype>(sz));     // store lambdaa(k), shape: (1,)
    for (int i = 0; i < 3; ++i)
    {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
  // Mask statistics from optimization by setting local learning rates
  // for mean, variance, and the bias correction to zero.
  for (int i = 0; i < this->blobs_.size(); ++i)
  {
    if (this->layer_param_.param_size() == i)
    {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
    }
    else
    {
      CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
          << "Cannot configure batch normalization statistics as layer "
          << "parameters.";
    }
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);                    // type: Blob<Dtype>, shape: (C,)
  variance_.Reshape(sz);                // type: Blob<Dtype>, shape: (C,)
  temp_.ReshapeLike(*bottom[0]);        // type: Blob<Dtype>, shape: (N,C,H,W)
  x_norm_.ReshapeLike(*bottom[0]);      // type: Blob<Dtype>, shape: (N,C,H,W)
  sz[0] = bottom[0]->shape(0);
  batch_sum_multiplier_.Reshape(sz);    // type: Blob<Dtype>, shape: (N,), all elements are Dtype(1)

  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim)
  {
    sz[0] = spatial_dim;
    // type: Blob<Dtype>, shape: (HW,), all elements are Dtype(1)
    spatial_sum_multiplier_.Reshape(sz);
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
  }

  int numbychans = channels_*bottom[0]->shape(0);
  if (num_by_chans_.num_axes() == 0 ||
      num_by_chans_.shape(0) != numbychans)
  {
    sz[0] = numbychans;
    // type: Blob<Dtype>, shape: (NC,)
    num_by_chans_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
              batch_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);

  if (bottom[0] != top[0])
  {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  if (use_global_stats_)
  {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ? 0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_cpu_scale(variance_.count(), scale_factor, this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
    caffe_cpu_scale(variance_.count(), scale_factor, this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
  }
  else
  {
    // M: channels_ * num = NC
    // N: spatial_dim = HW
    // alpha: 1. / (num * spatial_dim) = 1./(HW)
    // A: bottom_data, shape in math: (NC,HW)
    // x: spatial_sum_multiplier_.cpu_data(), shape in math: (HW,1), all elements are Dtype(1)
    // beta: 0.
    // y: num_by_chans_.mutable_cpu_data(), shape in math: (NC,1)
    // function: Compute EX|(NC,1), shape in math: (NC,1).
    // It is the mean of bottom_data with shape (N,C,H,W) along H and W axes.
    caffe_cpu_gemv<Dtype>(CblasNoTrans,
                          channels_ * num,
                          spatial_dim,
                          1. / (num * spatial_dim),
                          bottom_data,
                          spatial_sum_multiplier_.cpu_data(),
                          0.,
                          num_by_chans_.mutable_cpu_data());

    // M: num = N
    // N: channels_ = C
    // alpha: 1.
    // A: num_by_chans_.cpu_data(), is EX|(NC,1), shape in math: (N,C)
    // x: batch_sum_multiplier_.cpu_data(), shape in math: (N,1), all elements are Dtype(1)
    // beta: 0.
    // y: mean_.mutable_cpu_data(), shape in math: (C,1)
    // function: Compute EX|(C,1), namely mean of this batch data, shape in math: (C,1).
    caffe_cpu_gemv<Dtype>(CblasTrans,
                          num,
                          channels_,
                          1.,
                          num_by_chans_.cpu_data(),
                          batch_sum_multiplier_.cpu_data(),
                          0.,
                          mean_.mutable_cpu_data());
  }

  // M: num = N
  // N: channels_ = C
  // K: 1
  // alpha: 1
  // A: batch_sum_multiplier_.cpu_data(), shape in math: (N,1), all elements are Dtype(1)
  // B: mean_.cpu_data(), is EX|(C,1), shape in math: (1,C)
  // beta: 0.
  // C: num_by_chans_.mutable_cpu_data(), shape in math: (N,C)
  // function: Compute EX|(N,C), shape in math: (N,C).
  // It is a matrix with every row being an [EX|(C,1)]^T.
  caffe_cpu_gemm<Dtype>(CblasNoTrans,
                        CblasNoTrans,
                        num,
                        channels_,
                        1,
                        1,
                        batch_sum_multiplier_.cpu_data(),
                        mean_.cpu_data(),
                        0.,
                        num_by_chans_.mutable_cpu_data());

  // M: channels_ * num = NC
  // N: spatial_dim = HW
  // K: 1
  // alpha: -1
  // A: num_by_chans_.cpu_data(), is EX|(N,C), shape in math: (NC,1)
  // B: spatial_sum_multiplier_.cpu_data(), shape in math: (HW,1), all elements are Dtype(1)
  // beta: 1.
  // C: top_data, shape in math: (NC,HW)
  // function: Compute (X-EX)|(NC,HW) and store it in top_data, shape in math: (NC,HW).
  caffe_cpu_gemm<Dtype>(CblasNoTrans,
                        CblasNoTrans,
                        channels_ * num,
                        spatial_dim,
                        1,
                        -1,
                        num_by_chans_.cpu_data(),
                        spatial_sum_multiplier_.cpu_data(),
                        1.,
                        top_data);

  if (!use_global_stats_)
  {
    // Compute (X-EX)^2, namely D^2, shape in blob: (N,C,H,W).
    caffe_sqr<Dtype>(top[0]->count(), top_data, temp_.mutable_cpu_data());

    // Compute ED^2|(NC,1), shape in math: (NC,1)
    caffe_cpu_gemv<Dtype>(CblasNoTrans,
                          channels_ * num,
                          spatial_dim,
                          1. / (num * spatial_dim),
                          temp_.cpu_data(),
                          spatial_sum_multiplier_.cpu_data(),
                          0.,
                          num_by_chans_.mutable_cpu_data());

    // Compute ED^2|(C,1), namely variance this batch data, shape in math: (C,1).
    caffe_cpu_gemv<Dtype>(CblasTrans,
                          num,
                          channels_,
                          1.,
                          num_by_chans_.cpu_data(),
                          batch_sum_multiplier_.cpu_data(),
                          0.,
                          variance_.mutable_cpu_data());

    // lambdaa(k) = lambda * lambdaa(k-1) + 1
    this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    // mua(k) = lambda * mua(k-1) + mu(k)
    caffe_cpu_axpby(mean_.count(),
                    Dtype(1),
                    mean_.cpu_data(),
                    moving_average_fraction_,
                    this->blobs_[0]->mutable_cpu_data());
    int m = bottom[0]->count()/channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    // sigmaa^2(k) = sigmaa^2(k-1) + m/(m-1)*sigma^2(k)
    caffe_cpu_axpby(variance_.count(),
                    bias_correction_factor,
                    variance_.cpu_data(),
                    moving_average_fraction_,
                    this->blobs_[1]->mutable_cpu_data());
  }

  // normalize variance
  // sigma^2(k) = sigma^2(k) - eps
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  // sigma(k) = sqrt(sigma^2(k))
  caffe_sqrt(variance_.count(), variance_.cpu_data(),
             variance_.mutable_cpu_data());

  // replicate variance to input size
  caffe_cpu_gemm<Dtype>(CblasNoTrans,
                        CblasNoTrans,
                        num,
                        channels_,
                        1,
                        1,
                        batch_sum_multiplier_.cpu_data(),
                        variance_.cpu_data(),
                        0.,
                        num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans,
                        CblasNoTrans,
                        channels_ * num,
                        spatial_dim,
                        1,
                        1.,
                        num_by_chans_.cpu_data(),
                        spatial_sum_multiplier_.cpu_data(),
                        0.,
                        temp_.mutable_cpu_data());
  caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  // might clobber the data.  Can we skip this if they won't?
  caffe_copy(x_norm_.count(), top_data, x_norm_.mutable_cpu_data());
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->cpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
    top_diff = x_norm_.cpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (use_global_stats_) {
    caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
    return;
  }
  const Dtype* top_data = x_norm_.cpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());

  // reshape (broadcast) the above
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(BatchNormLayer);
#endif

INSTANTIATE_CLASS(BatchNormLayer);
REGISTER_LAYER_CLASS(BatchNorm);
}  // namespace caffe
