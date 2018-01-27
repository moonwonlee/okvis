#ifndef _CORRESPONDENCE_RANSAC_H_
#define _CORRESPONDENCE_RANSAC_H_

#include <vector>
#include <Eigen/Core>
#include "point_cloud.hpp"
#include <iostream>
//#include "time_profiler.hpp"

class CorrespondenceRansac{
public:

	///
	/// This function performs ransac on a set of feature
	/// correspondences to determince which are inliers
	/// also returns an estimate of the transform between clouds
	///
	static void getInliersWithTransform(
			const PointCloud& src, 
			const PointCloud& tgt,
			const std::vector<int>& correspondences,
			int num_samples, 
			std::vector<float> inlier_threshold, 
			int num_iterations,
			int& num_inliers_out,
			std::vector<bool>& inliers_out,
			Eigen::Affine3d& transform_out,
			bool exit_condition_flag=true,
			float exit_threshold=0.5){
		//MYPROFILE("RANSAC");

		//We need at least 3 samples to compute a transform
		if(num_samples<3){
			std::cerr << "Need at least 3 samples to perform RANSAC\n";
			std::cerr << "Setting num_samples to 3\n";
			num_samples =3;
		}

		//We don't know what was fed in with inliers_out so we should resize
		//and reset all values to false
		inliers_out.resize(correspondences.size());
		for(size_t i = 0; i<inliers_out.size(); i++){
			inliers_out[i] = false;
		}

		//Ensure correspondences is same size as src cloud
		if(src.size()!=correspondences.size()){
			std::cerr << 
				"There must be correspondence for each point in src cloud\n";
			std::cerr << 
				"src->size(): " << src.size() << " corr: " << correspondences.size() << '\n';

			transform_out = Eigen::Affine3d::Identity();
			num_inliers_out = 0;
			return;
		}

		//If we have less than the number of samples needed for ransac
		//just return identity transform
		if(inliers_out.size()<num_samples){
			std::cerr << "Not enough points for ransac\n";
			transform_out = Eigen::Affine3d::Identity();
			num_inliers_out = 0;
			return;
		}

		//Used for storing the best iteration result
		int best_inliers = 0;	        
    Eigen::Affine3d best_transform = Eigen::Affine3d::Identity();

    //This will be for transforming cloud
    //I think using PCL transform cloud will be faster than 
    //Applying transform to each point individually, despite extra memory usage
    //Should test eventually though, similarly using Eigen to store clouds may be faster
    PointCloud temp_cloud(correspondences.size());


    //Run RANSAC the specified number of times 
    for ( int r = 0; r< num_iterations; r++){
    	//Choose random points
    	std::vector<Eigen::Vector3d> s_points(num_samples);
    	std::vector<Eigen::Vector3d> t_points(num_samples);
    	
    	for( int i = 0; i<num_samples; i++){

    		int index(rand()%src.size());
    		s_points[i] = src[index];
    		t_points[i] = tgt[correspondences[index]];
    	}




		Eigen::Matrix3d W(Eigen::Matrix3d::Zero()); //point correspondence matrix used to compute R

		//compute mean points
		Eigen::Vector3d s_mu(Eigen::Vector3d::Zero()), t_mu(Eigen::Vector3d::Zero());
		for( int i = 0; i<num_samples; i++){
			s_mu+=s_points[i];
			t_mu+=t_points[i];
		}
		s_mu/=num_samples;
		t_mu/=num_samples;



		//compute W as W = sum(s_x*t_x.transpose())
		for( int i = 0; i<num_samples; i++){
			W+=(s_points[i]-s_mu)*(t_points[i]-t_mu).transpose();
		}
		//std::cout << "W:= " << W << '\n';
		//compute transform from SVD
		Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
		//std::cout << "rank: " << svd.rank() << '\n';
		//ensure W was full rank
		//if(svd.rank()<3)
		//	continue;
		Eigen::Quaterniond R((svd.matrixU()*svd.matrixV().transpose()).transpose());
		Eigen::Translation3d t(t_mu - R*s_mu);
		Eigen::Affine3d transform = t*R;

		//apply transform to cloud
		transform_point_cloud(src,temp_cloud,transform);


		//count inliers
      int inliers=0; 
      for (size_t i = 0; i < src.size(); ++i)
      {
        	inliers+=checkThreshold(temp_cloud[i],tgt[correspondences[i]],inlier_threshold[i]);
      }


      //save best result
      if(inliers>best_inliers){
      	best_inliers=inliers;
      	best_transform = transform;
      }    
      if(exit_condition_flag && (float)best_inliers/correspondences.size()>exit_threshold)
      	break;

    }
    //Fill in output with best result
    transform_out = best_transform;
    transform_point_cloud(src,temp_cloud,best_transform);
    num_inliers_out= 0;
   	for (size_t i = 0; i < src.size(); ++i)
    {
    	num_inliers_out+=inliers_out[i] = checkThreshold(temp_cloud[i],tgt[correspondences[i]],inlier_threshold[i]);
    }



  }	


static void getFinalTransform(
      const PointCloud& src, 
      const PointCloud& tgt,
      const std::vector<int>& correspondences,
      const std::vector<bool>& inliers,
      Eigen::Affine3d& transform_out){
    //MYPROFILE("RANSAC");

    //Ensure correspondences is same size as src cloud
    if(src.size()!=correspondences.size()){
      std::cerr << 
        "There must be correspondence for each point in src cloud\n";
      std::cerr << 
        "src->size(): " << src.size() << " corr: " << correspondences.size() << '\n';

      transform_out = Eigen::Affine3d::Identity();
      return;
    }

    //If we have less than the number of samples needed for ransac
    //just return identity transform
    if(inliers.size()<src.size()){
      std::cerr << 
        "There must be inlier flag for each point in src cloud\n";
      transform_out = Eigen::Affine3d::Identity();
      return;
    }


    int num_inliers=0;

    Eigen::Matrix3d W(Eigen::Matrix3d::Zero()); //point correspondence matrix used to compute R

    //compute mean points
    Eigen::Vector3d s_mu(Eigen::Vector3d::Zero()), t_mu(Eigen::Vector3d::Zero());
    for( int i = 0; i<src.size(); i++){
      //5.0 is max trusted depth.
      //This method has a problem of not taking into account
      //the certainty of the points given in the clouds
      //should ideally be replaced with convex opt approach
      if(inliers[i] && src[i].norm()<5.0){
        s_mu+=src[i];
        t_mu+=tgt[correspondences[i]];
        num_inliers++;
      }
    }
    s_mu/=num_inliers;
    t_mu/=num_inliers;



    //compute W as W = sum(s_x*t_x.transpose())
    for( int i = 0; i<src.size(); i++){
      if(inliers[i]){
        W+=(src[i]-s_mu)*(tgt[correspondences[i]]-t_mu).transpose();
      }
    }
    //std::cout << "W:= " << W << '\n';
    //compute transform from SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    //std::cout << "rank: " << svd.rank() << '\n';
    //ensure W was full rank
    //if(svd.rank()<3)
    //  continue;
    Eigen::Quaterniond R((svd.matrixU()*svd.matrixV().transpose()).transpose());
    Eigen::Translation3d t(t_mu - R*s_mu);
    transform_out = t*R;

} 



private:

	///
	/// Checks transformed points against threshold
	///
	static bool checkThreshold(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, float inlier_threshold){
		return ((p1.x()-p2.x())*(p1.x()-p2.x())+
				(p1.y()-p2.y())*(p1.y()-p2.y())+
				(p1.z()-p2.z())*(p1.z()-p2.z()))<
				(inlier_threshold*inlier_threshold);
	}

	CorrespondenceRansac(){}
};



#endif


