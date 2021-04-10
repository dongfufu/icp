#include <iostream>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <vector>
using namespace std;

// Read a Bundle Adjustment in the Large dataset.
// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
//残差块,输入两组点
//四元数存储输出按照x,y,z,w来做，初始化按照w,x,y,z来做

struct   ICPError {
  ICPError(Eigen::Vector3d  par_point,Eigen::Vector3d child_point){
    par_point_=par_point;
    child_point_=child_point;
}
 template<typename T>
 bool operator()(const T* q_res,const T *const t_res,  T*residuals ) const{//transform为待优化变量，有6维   
    //q_res为一个存储四元数的数组
    //设置四元数,构造w,x,y,z ,存储和输出x,y,z,w
     Eigen::Quaternion<T> q_use(q_res[3],q_res[0],q_res[1],q_res[2]);
     Eigen::Matrix<T,3,1> t_use(t_res[0],t_res[1],t_res[2]);
     Eigen::Matrix<T,3,1> m_par_point(T(par_point_(0)),T(par_point_(1)),T(par_point_(2)));
     Eigen::Matrix<T,3,1> m_child_point(T(child_point_(0)),T(child_point_(1)),T(child_point_(2)));
     Eigen::Matrix<T,3,1> my_after_child_point=q_use*m_child_point+t_use;
    //定义残差
    residuals[0]=m_par_point(0)-my_after_child_point(0);
    residuals[1]=m_par_point(1)-my_after_child_point(1);
    residuals[2]=m_par_point(2)-my_after_child_point(2);
    return true;
}
  static  ceres::CostFunction*  Create_Cost_Fun(const Eigen::Vector3d  par_point,Eigen::Vector3d  child_point)
  {
      //损失函数类型、输出维度、输入维度，此处输出维度为3，输入维度主要包括两个变量，四元数变量为4，平移变量维度为3
      return (new ceres::AutoDiffCostFunction<ICPError,3,4,3>(new ICPError(par_point,child_point)));
}
Eigen::Vector3d  par_point_,child_point_;
};





 int main(){
     std::vector<Eigen::Vector3d> par_points;
     std::vector<Eigen::Vector3d> child_points;
//初始化子点云
     for(int i=0;i<200;i++)
     {
          Eigen::Vector3d  temp(i,i+1,i+2);
          cout<<"temp="<<temp<<endl;
          child_points.push_back(temp);
    }
 
 //定义父到子之间的旋转关系
  // Eigen::Matrix3d  R;
   //wxyz初始化，xyzw存储
   Eigen::Quaternion<double> q(1,0.2,0.2,0.2);
   q.normalize();
   Eigen::Vector3d t;
//     //eigen中默认按行输入
  //  R<<1 ,0 ,0 ,0, 1, 0, 0, 0, 1; 
     t<<2,3,4;
     //根据转换关系生成父点云
    for(int i=0;i<200;i++)
     {
         Eigen::Vector3d   par_point=q*child_points[i]+t;
           cout<<"par_point="<<par_point<<endl;
         par_points.push_back(par_point);
    }
    

   //设置优化初值
   double  q_res[4]={0,0,0,0.5};
   double t_res[3]={1,2,3};
   //开始优化
   ceres::Problem problem;
   for(int i=0;i<par_points.size();i++)
   {
       //调用构造体中的静态函数，创建损失函数
       ceres::CostFunction*cost=ICPError::Create_Cost_Fun(par_points[i],child_points[i]);
       problem.AddResidualBlock(cost,NULL,q_res,t_res);
  }
  //设置优化参数，开始优化
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations=500;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
   cout<<"优化后的平移量"<<t_res[0]<<" "<<t_res[1]<<" "<<t_res[2]<<endl;
    cout<<"正确的平移量"<<t(0)<<" "<<t(1)<<" "<<t(2)<<endl;
    Eigen::Vector3d tt(t_res[0],t_res[1],t_res[2]);
   cout<<q_res[0]<<" "<<q_res[1]<<" "<<q_res[2]<<" "<<q_res[3]<<endl;
   Eigen::Quaternion<double> qq(q_res[3],q_res[0],q_res[1],q_res[2]);
   qq.normalize();
   cout<<"优化后的四元数"<<" "<<qq.coeffs()<<endl ;
   cout<<"真实四元数"<<" "<<q.coeffs()<<endl;
   // ceres::AngleAxisToRotationMatrix(aix,RR);
//根据优化后的参数计算点云重投影误差
    double sum_error=0;
    double  mse=0;//标准差
    for(int i=0;i<child_points.size();i++)
    {
        Eigen::Vector3d  tran_child_point=qq*child_points[i]+tt;
        sum_error+=(pow(tran_child_point(0)-par_points[i](0),2)+pow(tran_child_point(1)-par_points[i](1),2)+pow(tran_child_point(2)-par_points[i](2),2));
    }
    mse=sqrt(sum_error/child_points.size());
    cout<<"误差"<<mse<<endl;
        
     return 0;
}



