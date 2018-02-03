#ifndef INCLUDE_OKVIS_POSE_GRAPH_HPP_
#define INCLUDE_OKVIS_POSE_GRAPH_HPP_

#include <okvis/assert_macros.hpp>

#include <okvis/Parameters.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/FrameTypedefs.hpp>
#include "point_cloud.hpp"
#include "correspondence_ransac.hpp"
#include "glog/logging.h"
#include "pose_graph_3d_error_term.h"
#include "types.h"
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/eigen.hpp>


#define RANSAC_POINTS 3 //Number of random samples per iteration
#define RANSAC_THRESHOLD .2f //Allowable distance in m to be inlier for object 1m away
#define RANSAC_ITERATIONS 100 //Number of ransac iterations to perform



#include <DBoW2.h>

/// \brief okvis Main namespace of this package.
namespace okvis {

inline bool in_image(cv::Point pt, const cv::Mat& image){
  return pt.y>0 && pt.y < image.rows && pt.x>0 && pt.x < image.cols;
}

///
/// Converts a vector between cv::Point2f and cv::Keypoint
/// Needed for different algorithms in OpenCV
///
inline void keypoints_to_features(const std::vector<cv::KeyPoint>& keypoints_in, 
                  std::vector<cv::Point2f>& features_out){
  features_out.resize(keypoints_in.size());
  for(int i =0; i<keypoints_in.size(); i++){
    features_out[i] = keypoints_in[i].pt;
  }

}



/**
 * @brief This class is responsible to visualize the matching results
 */
class PoseGraph {
 public:

  PoseGraph(std::string vocabPath):
  posesSinceLastLoop_(0),
  lastKeyframeT_SoW(Eigen::Matrix4d::Identity()),
  currentKeyframeT_WSo(Eigen::Matrix4d::Identity()){
      std::cout << "Loading Vocabulary From: " << vocabPath << std::endl;
      vocab_.reset(new OrbVocabulary(vocabPath));
      std::cout << "Vocabulary Size: " << vocab_->size() << std::endl;
      db_.reset(new OrbDatabase(*vocab_));
  }

  void BuildOptimizationProblem(::ceres::Problem* problem){
    BuildOptimizationProblem(constraints_,&nodes_,gravity_,problem);
  }


  // Constructs the nonlinear least squares optimization problem from the pose
  // graph constraints.
  void BuildOptimizationProblem(const VectorOfConstraints& constraints,
                                MapOfPoses* poses, const VectorOfGravityConstraints& gravity_nodes,
                                ::ceres::Problem* problem) {
    CHECK(poses != NULL);
    CHECK(problem != NULL);
    if (constraints.empty()) {
      LOG(INFO) << "No constraints, no problem to optimize.";
      return;
    }

    ::ceres::LossFunction* loss_function = NULL;
    ::ceres::LocalParameterization* quaternion_local_parameterization =
        new ::ceres::EigenQuaternionParameterization;

    for (VectorOfConstraints::const_iterator constraints_iter =
             constraints.begin();
         constraints_iter != constraints.end(); ++constraints_iter) {
      const Constraint3dNode& constraint = *constraints_iter;

      MapOfPoses::iterator pose_begin_iter = poses->find(constraint.id_begin);
      CHECK(pose_begin_iter != poses->end())
          << "Pose with ID: " << constraint.id_begin << " not found.";
      MapOfPoses::iterator pose_end_iter = poses->find(constraint.id_end);
      CHECK(pose_end_iter != poses->end())
          << "Pose with ID: " << constraint.id_end << " not found.";

      const Eigen::Matrix<double, 6, 6> sqrt_information =
          constraint.information.llt().matrixL();
      // Ceres will take ownership of the pointer.
      ::ceres::CostFunction* cost_function =
          PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

      problem->AddResidualBlock(cost_function, loss_function,
                                pose_begin_iter->second.p.data(),
                                pose_begin_iter->second.q.coeffs().data(),
                                pose_end_iter->second.p.data(),
                                pose_end_iter->second.q.coeffs().data());

      problem->SetParameterization(pose_begin_iter->second.q.coeffs().data(),
                                   quaternion_local_parameterization);
      problem->SetParameterization(pose_end_iter->second.q.coeffs().data(),
                                   quaternion_local_parameterization);
    }


    for (VectorOfGravityConstraints::const_iterator gravity_iter =
             gravity_nodes.begin();
         gravity_iter != gravity_nodes.end(); ++gravity_iter) {
      GravityNode gravity = *gravity_iter;
      MapOfPoses::iterator pose_iter = poses->find(gravity.id);
      CHECK(pose_iter != poses->end())
          << "Pose with ID: " << gravity.id << " not found.";

      const Eigen::Matrix<double, 3, 3> sqrt_information = gravity.information.llt().matrixL();

      ::ceres::CostFunction* cost_function =
          PoseGraphGravityTerm::Create(gravity.g, sqrt_information);

      problem->AddResidualBlock(cost_function, loss_function,
                                pose_iter->second.q.coeffs().data());

      problem->SetParameterization(pose_iter->second.q.coeffs().data(),
                                   quaternion_local_parameterization);


    }

    // The pose graph optimization problem has six DOFs that are not fully
    // constrained. This is typically referred to as gauge freedom. You can apply
    // a rigid body transformation to all the nodes and the optimization problem
    // will still have the exact same cost. The Levenberg-Marquardt algorithm has
    // internal damping which mitigates this issue, but it is better to properly
    // constrain the gauge freedom. This can be done by setting one of the poses
    // as constant so the optimizer cannot change it.
    MapOfPoses::iterator pose_start_iter = poses->begin();
    CHECK(pose_start_iter != poses->end()) << "There are no poses.";
    problem->SetParameterBlockConstant(pose_start_iter->second.p.data());
    problem->SetParameterBlockConstant(pose_start_iter->second.q.coeffs().data());
  }

  // Returns true if the solve was successful.
  bool SolveOptimizationProblem(::ceres::Problem* problem) {
    CHECK(problem != NULL);

    ::ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ::ceres::SPARSE_NORMAL_CHOLESKY;

    ::ceres::Solver::Summary summary;
    ::ceres::Solve(options, problem, &summary);

    std::cout << summary.FullReport() << '\n';

    return summary.IsSolutionUsable();
  }

  bool OutputPoses(const std::string& filename){
    return OutputPoses(filename,nodes_);
  }

  // Output the poses to the file with format: id x y z q_x q_y q_z q_w.
  bool OutputPoses(const std::string& filename, const MapOfPoses& poses) {
    std::fstream outfile;
    outfile.open(filename.c_str(), std::istream::out);
    if (!outfile) {
      LOG(ERROR) << "Error opening the file: " << filename;
      return false;
    }
    for (std::map<int, Pose3dNode, std::less<int>,
                  Eigen::aligned_allocator<std::pair<const int, Pose3dNode> > >::
             const_iterator poses_iter = poses.begin();
         poses_iter != poses.end(); ++poses_iter) {
      const std::map<int, Pose3dNode, std::less<int>,
                     Eigen::aligned_allocator<std::pair<const int, Pose3dNode> > >::
          value_type& pair = *poses_iter;
      outfile << pair.first << " " << pair.second.p.transpose() << " "
              << pair.second.q.x() << " " << pair.second.q.y() << " "
              << pair.second.q.z() << " " << pair.second.q.w() << '\n';
    }
    return true;
  }

  /// @brief This struct contains the relevant data for visualizing
  struct KeyframeData {
    typedef std::shared_ptr<KeyframeData> Ptr;
    std::shared_ptr<okvis::MultiFrame> keyFrames;     ///< Current keyframe.
    okvis::kinematics::Transformation T_WS;  ///< Pose of the current keyframe
    okvis::kinematics::Transformation T_SoSn;  ///< Pose of the current keyframe
    okvis::ObservationVector observations;
    DBoW2::EntryId id;
    DBoW2::BowVector bowVec;
    cv::Mat descriptors;
  };

  std::unique_ptr<OrbVocabulary> vocab_;
  std::unique_ptr<OrbDatabase> db_;
  std::vector<KeyframeData::Ptr> poses_;

  okvis::kinematics::Transformation lastKeyframeT_SoW;
  okvis::kinematics::Transformation currentKeyframeT_WSo;

  size_t posesSinceLastLoop_;
  DBoW2::EntryId lastEntry_;

  VectorOfConstraints constraints_;
  VectorOfGravityConstraints gravity_;
  MapOfPoses nodes_;
  MapOfPoses originalNodes_;

private: 

 };


} //namespace okvis


#endif //INCLUDE_OKVIS_POSE_GRAPH_HPP_