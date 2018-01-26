#ifndef _POINT_CLOUD_H_
#define _POINT_CLOUD_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <opencv2/core/core.hpp>
#include <memory>


typedef std::vector<Eigen::Vector3d> PointCloud ;

///
///	Transforms all points in cloud using given transform
///
inline void transform_point_cloud(const PointCloud& in_cloud, PointCloud& out_cloud,
						const Eigen::Affine3d& transform) {
  out_cloud.clear();
  out_cloud.reserve(in_cloud.size());
  for (const Eigen::Vector3d& point : in_cloud) {
    out_cloud.emplace_back(transform * point);
  }
}

///
///	Projects points to camera using given transform and intrinsics matrix 
///
inline void proj_points_to_camera(const PointCloud& cloud_in, 
						std::vector<cv::Point2f>& features_out, 
						const Eigen::Matrix3d& intrinsics_in, 
						const Eigen::Affine3d& transform_in){

	//Transform keyframe point cloud using previous estimate
	PointCloud temp_cloud;
	transform_point_cloud(cloud_in, temp_cloud,
				transform_in);

	//Apply camera projection 
	features_out.resize(temp_cloud.size());
	
	for(size_t i=0; i < temp_cloud.size(); i++){
		temp_cloud[i] = intrinsics_in*temp_cloud[i]/temp_cloud[i].z();
		features_out[i] = cv::Point2f(temp_cloud[i].x(),temp_cloud[i].y());
	}
}

///
///	Computes 3d positions of points in camera frame using inverse intrinsics matrix
///
inline void cloud_from_depth(const std::vector<cv::Point2f>& features_in, 
						const std::vector<float>& feature_depths_in, 
						const Eigen::Matrix3d& inv_intrinsics_in, 
						PointCloud& cloud_out){

		cloud_out.resize(features_in.size());
		for(size_t i=0; i < features_in.size(); i++){
		/*convert point to 3d coordinates:
		 X_w = K^-1*x_h*depth
		 where: X_w is point in 3d coords
		 		x_h is point in homogeneous camera coords
		 		K is the camera calibration matrix
		 		depth is the depth value of the point */

		cloud_out[i] = inv_intrinsics_in*
			Eigen::Vector3d(features_in[i].x,features_in[i].y,1)*
			feature_depths_in[i];

	}
}


#endif
