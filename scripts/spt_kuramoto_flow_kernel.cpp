#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>

// Float64 RK4 and its source variational equation. No fast-math, no changed
// force law or stochastic approximation. The Python implementation is retained
// for parameter sensitivities and independent numerical agreement tests.
static void rhs(const double* x, const double* jac, double* force, double* dj,
                const int* edges, const int* sizes, int count,
                const double* weights, const double* frequency) {
    double derivative[81] = {};
    std::copy(frequency, frequency+9, force);
    for (int e=0; e<count; ++e) {
        const int* nodes=edges+9*e;
        const int m=sizes[e];
        double sum=0;
        for (int j=0; j<m; ++j) sum+=x[nodes[j]];
        for (int j=0; j<m; ++j) {
            const int i=nodes[j];
            const double angle=sum-m*x[i];
            force[i]+=weights[e]*std::sin(angle);
            if (jac) {
                const double c=weights[e]*std::cos(angle);
                for (int k=0; k<m; ++k) derivative[9*i+nodes[k]]+=c;
                derivative[9*i+i]-=m*c;
            }
        }
    }
    if (jac) {
        for (int i=0; i<9; ++i) for (int j=0; j<9; ++j) {
            double sum=0;
            for (int k=0; k<9; ++k) sum+=derivative[9*i+k]*jac[9*k+j];
            dj[9*i+j]=sum;
        }
    }
}

extern "C" void kuramoto_flow(int n, const double* source, double* output, double* output_jac,
    const int* edges, const int* sizes, int count, const double* weights,
    const double* frequency, double horizon, int steps, int thread_count) {
    auto work = [&](int first,int last) {
        const double h=horizon/steps;
        for (int row=first; row<last; ++row) {
            double x[9], tmp[9], k1[9], k2[9], k3[9], k4[9];
            double jac[81]={}, jtmp[81], d1[81], d2[81], d3[81], d4[81];
            std::copy(source+9*row,source+9*row+9,x);
            for (int i=0; i<9; ++i) jac[9*i+i]=1;
            for (int step=0; step<steps; ++step) {
                rhs(x,output_jac?jac:nullptr,k1,d1,edges,sizes,count,weights,frequency);
                for (int i=0; i<9; ++i) tmp[i]=x[i]+h*k1[i]/2;
                if (output_jac) for (int i=0; i<81; ++i) jtmp[i]=jac[i]+h*d1[i]/2;
                rhs(tmp,output_jac?jtmp:nullptr,k2,d2,edges,sizes,count,weights,frequency);
                for (int i=0; i<9; ++i) tmp[i]=x[i]+h*k2[i]/2;
                if (output_jac) for (int i=0; i<81; ++i) jtmp[i]=jac[i]+h*d2[i]/2;
                rhs(tmp,output_jac?jtmp:nullptr,k3,d3,edges,sizes,count,weights,frequency);
                for (int i=0; i<9; ++i) tmp[i]=x[i]+h*k3[i];
                if (output_jac) for (int i=0; i<81; ++i) jtmp[i]=jac[i]+h*d3[i];
                rhs(tmp,output_jac?jtmp:nullptr,k4,d4,edges,sizes,count,weights,frequency);
                for (int i=0; i<9; ++i) x[i]+=h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;
                if (output_jac) for (int i=0; i<81; ++i) jac[i]+=h*(d1[i]+2*d2[i]+2*d3[i]+d4[i])/6;
            }
            std::copy(x,x+9,output+9*row);
            if (output_jac) std::copy(jac,jac+81,output_jac+81*row);
        }
    };
    const int threads=std::max(1,std::min(thread_count,n));
    std::vector<std::thread> workers;
    for (int t=1; t<threads; ++t) workers.emplace_back(work,n*t/threads,n*(t+1)/threads);
    work(0,n/threads);
    for (auto& worker:workers) worker.join();
}
