#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>
#include <algorithm>
#include <atomic>

// Algebraically identical to exp(u @ v) for binary-subset vectors v=h/pi.
// No fast-math, approximated exponentials, reduced precision or true supports.
extern "C" int glv_mean(long n,const double* source,const double* bias,
    int terms,const int32_t* masks,const int32_t* powers,const int32_t* outputs,
    const double* coefficients,double* mean,double* jacobian,int thread_count) {
  const double pi=std::acos(-1.0);
  std::atomic<int> error(0);
  auto work=[&](long first,long last) {
  for(long row=first;row<last;++row) {
    const double* x=source+9*row;
    double* y=mean+9*row;
    double* jac=jacobian ? jacobian+81*row : nullptr;
    for(int i=0;i<9;++i) y[i]=x[i]+bias[i];
    if(jac) for(int i=0;i<81;++i) jac[i]=(i/9==i%9) ? 1.0 : 0.0;
    if(!terms) continue;
    double base[3][9];
    for(int j=0;j<9;++j) {
      base[0][j]=std::exp(x[j]/pi);
      base[1][j]=base[0][j]*base[0][j];
      base[2][j]=base[1][j]*base[0][j];
    }
    double products[3][512];
    for(int p=0;p<3;++p) {
      products[p][0]=1.0;
      for(unsigned mask=1;mask<512;++mask) {
        unsigned low=mask & (~mask+1u);
        int j=__builtin_ctz(low);
        products[p][mask]=products[p][mask^low]*base[p][j];
      }
    }
    int term=0;
    for(int i=0;i<9;++i) {
      double sum=y[i];
      while(term<terms && outputs[term]==i) {
        double value=coefficients[term]*products[powers[term]-1][masks[term]];
        sum+=value;
        if(jac) {
          double derivative=value*powers[term]/pi;
          unsigned bits=masks[term];
          while(bits) {int j=__builtin_ctz(bits); jac[i*9+j]+=derivative; bits&=bits-1;}
        }
        ++term;
      }
      y[i]=sum;
    }
    for(int i=0;i<9;++i) if(!std::isfinite(y[i])) {error.store(1); return;}
  }
  };
  const int threads=std::max(1,std::min(thread_count,static_cast<int>(n)));
  std::vector<std::thread> workers;
  for(int t=1;t<threads;++t) workers.emplace_back(work,n*t/threads,n*(t+1)/threads);
  work(0,n/threads);
  for(auto& worker:workers) worker.join();
  return error.load();
}

// Selected outputs and selected source derivatives of exactly the same mean.
extern "C" int glv_subset(long n,const double* source,const double* bias,
    int terms,const int32_t* masks,const int32_t* powers,const int32_t* outputs,
    const double* coefficients,int nt,const int32_t* targets,
    int nd,const int32_t* variables,double* mean,double* jacobian,int thread_count) {
  const double pi=std::acos(-1.0);
  int starts[10]={0};
  for(int k=0;k<terms;++k) ++starts[outputs[k]+1];
  for(int k=1;k<10;++k) starts[k]+=starts[k-1];
  std::atomic<int> error(0);
  auto work=[&](long first,long last) {
    for(long row=first;row<last;++row) {
      const double* x=source+9*row;
      double products[3][512];
      if(terms) {
        double base[3][9];
        for(int j=0;j<9;++j) {
          base[0][j]=std::exp(x[j]/pi);
          base[1][j]=base[0][j]*base[0][j];
          base[2][j]=base[1][j]*base[0][j];
        }
        for(int p=0;p<3;++p) {
          products[p][0]=1.;
          for(unsigned mask=1;mask<512;++mask) {
            unsigned low=mask & (~mask+1u);
            products[p][mask]=products[p][mask^low]*base[p][__builtin_ctz(low)];
          }
        }
      }
      for(int t=0;t<nt;++t) {
        const int i=targets[t];
        double sum=x[i]+bias[i];
        double* jac=jacobian ? jacobian+(row*nt+t)*nd : nullptr;
        if(jac) for(int v=0;v<nd;++v) jac[v]=(i==variables[v]) ? 1. : 0.;
        for(int k=starts[i];k<starts[i+1];++k) {
          double value=coefficients[k]*products[powers[k]-1][masks[k]];
          sum+=value;
          if(jac) {
            const double derivative=value*powers[k]/pi;
            for(int v=0;v<nd;++v) if(masks[k]&(1<<variables[v])) jac[v]+=derivative;
          }
        }
        mean[row*nt+t]=sum;
        if(!std::isfinite(sum)) {error.store(1); return;}
        if(jac) for(int v=0;v<nd;++v) if(!std::isfinite(jac[v])) {error.store(1);return;}
      }
    }
  };
  const int threads=std::max(1,std::min(thread_count,static_cast<int>(n)));
  std::vector<std::thread> workers;
  for(int t=1;t<threads;++t) workers.emplace_back(work,n*t/threads,n*(t+1)/threads);
  work(0,n/threads);
  for(auto& worker:workers) worker.join();
  return error.load();
}
