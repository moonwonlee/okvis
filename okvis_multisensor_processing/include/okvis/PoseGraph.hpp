#ifndef INCLUDE_OKVIS_POSE_GRAPH_HPP_
#define INCLUDE_OKVIS_POSE_GRAPH_HPP_

#include <okvis/assert_macros.hpp>

#include <okvis/Parameters.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/FrameTypedefs.hpp>
#include "glog/logging.h"
#include "pose_graph_3d_error_term.h"
#include "gravity_error_term.h"
#include "types.h"
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/eigen.hpp>


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

  PoseGraph(std::string vocabPath, okvis::VioParameters& parameters):
  parameters_(parameters),
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
  struct KeyFrameData {
    typedef std::shared_ptr<KeyFrameData> Ptr;
    std::shared_ptr<okvis::MultiFrame> keyFrames;     ///< Current keyframe.
    okvis::kinematics::Transformation T_WS;  ///< Pose of the current keyframe
    okvis::kinematics::Transformation T_SoSn;  ///< Pose of the current keyframe
    okvis::ObservationVector observations;
    DBoW2::EntryId id;
    DBoW2::BowVector bowVec;
    cv::Mat descriptors;
  };

  void processKeyFrame(KeyFrameData::Ptr kf, std::vector<okvis::kinematics::Transformation>& path_out, bool& loopClosure){

    loopClosure=false;
    //compute most optimized location of the current keyframe
    currentKeyframeT_WSo = currentKeyframeT_WSo*kf->T_SoSn;

    //This is only for debugging purposes
    Pose3dNode node;
    node.q = kf->T_WS.q();
    node.p = kf->T_WS.r();
    originalNodes_[poses_.size()]=node;
    
    //Update the new keyframe pose using thew best estimate from the optimized pose graph
    //kf->T_WS = currentKeyframeT_WSo;
    node.q = currentKeyframeT_WSo.q();
    node.p = currentKeyframeT_WSo.r();
    //Add this node to the pose graph
    nodes_[poses_.size()]=node;

    //As long as this is not the first pose
    if(poses_.size()>0 && parameters_.loopClosureParameters.enabled){
      //add constraint to pose graph between this pose and previous pose
      Pose3dNode newDiffConstraintNode;
      newDiffConstraintNode.q = kf->T_SoSn.q();
      newDiffConstraintNode.p = kf->T_SoSn.r();

      Constraint3dNode newDiffConstraint;
      newDiffConstraint.t_be = newDiffConstraintNode;
      newDiffConstraint.id_begin = poses_.size()-1;
      newDiffConstraint.id_end = poses_.size();
      newDiffConstraint.information = Eigen::Matrix<double,6,6>::Identity();
      newDiffConstraint.information.block<3,3>(3,3) = Eigen::Matrix<double,3,3>::Identity()*100;
      constraints_.push_back(newDiffConstraint);

      //I also include a "gravity constraint" 
      //This represents the fact that the tilt and roll estimates are bounded due to 
      //the presence of an accelerometer
      //---This may have a bug, disabling until a test can confirm
      /*GravityNode newGravityConstraint;
      newGravityConstraint.g = kf->T_WS.q().inverse()*Eigen::Vector3d(0,0,1);
      newGravityConstraint.id = poses_.size();
      newGravityConstraint.information = Eigen::Matrix3d::Identity();
      gravity_.push_back(newGravityConstraint);*/

    }

    //These are for debugging purposes
    //OutputPoses("poses.txt");
    //OutputPoses("orig_poses.txt", originalNodes_);
    path_out.clear();
    for (std::map<int, Pose3dNode, std::less<int>,
                  Eigen::aligned_allocator<std::pair<const int, Pose3dNode> > >::
             const_iterator poses_iter = nodes_.begin();
         poses_iter != nodes_.end(); ++poses_iter) {
      const std::map<int, Pose3dNode, std::less<int>,
                     Eigen::aligned_allocator<std::pair<const int, Pose3dNode> > >::
          value_type& pair = *poses_iter;
      path_out.push_back(okvis::kinematics::Transformation(pair.second.p,pair.second.q));  
    }

    if(!parameters_.loopClosureParameters.enabled)
      return;

    //Get feature points  
    std::vector<cv::KeyPoint> points;
    points.reserve(kf->observations.size());
    for (size_t k = 0; k < kf->observations.size(); ++k) {
      cv::KeyPoint kp;
      kf->keyFrames->getCvKeypoint(0,kf->observations[k].keypointIdx,kp);

      if(!okvis::in_image(kp.pt,kf->keyFrames->image(0))){
        std::cout << "Point Not in Image\n";
      }
      points.emplace_back(kp);
    }

    //detect orb descriptors
    //later will just use brisk descriptors, but need to create proper DBoW2 templates first
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->compute(kf->keyFrames->image(0),points,kf->descriptors);

    //if we can not compute orb descriptors for all points, do not use keyframe
    //this will not be an issue once we switch to brisk descriptors
    if(points.size()!=kf->observations.size())
      return;

    //convert to DBoW descriptor format
    std::vector<cv::Mat> bowDesc;
    bowDesc.reserve(kf->descriptors.rows);
    for(int i=0; i<kf->descriptors.rows; i++){
      bowDesc.emplace_back(kf->descriptors.row(i));
    } 

    //get BoW vector for keyframe 
    vocab_->transform(bowDesc, kf->bowVec);
    if(kf->bowVec.size()==0)
      return;
    if(posesSinceLastLoop_<20){
      //20 is the min number of keyframes before we want to consider a loop closure
      //this prevents us from trying to close the loop wuth very nearby frames
      lastEntry_ = kf->id = db_->add(kf->bowVec);
      poses_.push_back(kf);
      posesSinceLastLoop_++;
      return;
    }

    DBoW2::QueryResults qret;
    //get best result from database. Don't consider the 20 most recent images
    //should experiment with getting top 3 or top 5 results from database
    db_->query(kf->bowVec,qret,1,lastEntry_-20);
    //compute similarity score to previous image
    //this is necessary as the score will depend on the number and type of features in the image
    //see DBoW2 paper for more details
    float baseScore = vocab_->score(kf->bowVec,poses_[lastEntry_]->bowVec);
    if(qret.size()>0 && baseScore>0.1 && qret[0].Score/baseScore>0.75){ 
      //.01 is min similarity to previous frame,
      //.75 is required similarity of matched frame relative to previous frame
      //that is, the matched frame must be at least 75% as similar to the current
      //frame as the current frame is to the previous frame

      //if the matched frame is similar enough, attempt to compute SE3 transform
      PoseGraph::KeyFrameData::Ptr matchedFrame = poses_[qret[0].Id];

      //get keypoints from frame
      std::vector<cv::KeyPoint> matchedPoints;
      matchedPoints.reserve(matchedFrame->observations.size());
      for (size_t k = 0; k < matchedFrame->observations.size(); ++k) {
        cv::KeyPoint kp;
        matchedFrame->keyFrames->getCvKeypoint(0,matchedFrame->observations[k].keypointIdx,kp);

        if(!okvis::in_image(kp.pt,matchedFrame->keyFrames->image(0))){
          std::cout << "Point Not in Image\n";
        }
        matchedPoints.emplace_back(kp);
      }

      //match them to the current frame using optical flow 
      std::vector<cv::Point2f> matchedFeatures, newFeatures, newFeaturesFinal;
      okvis::keypoints_to_features(matchedPoints, matchedFeatures);

      std::vector<unsigned char> valid;
      std::vector<float> err;

      calcOpticalFlowPyrLK(
        matchedFrame->keyFrames->image(0), kf->keyFrames->image(0),
        matchedFeatures, newFeatures,valid,err,cv::Size(11,11),3,
        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 1e-4);

      //get the 3d position of each matched point
      //and create new feature vector that only uses the points that were matched
      std::vector<cv::Point3f> matchedPoints3d;
      for (size_t k = 0; k < matchedFrame->observations.size(); ++k) {
        if(valid[k]){
          Eigen::Vector4d matchedPoint4d = matchedFrame->observations[k].landmark_C;
          cv::Point3f matchedPoint3d(matchedPoint4d[0],matchedPoint4d[1],matchedPoint4d[2]);
          matchedPoints3d.push_back(matchedPoint3d);
          newFeaturesFinal.push_back(newFeatures[k]);
        }
      }

      //get intrinsics and distortion coeffs from fame
      Eigen::VectorXd full_intrinsics;
      kf->keyFrames->geometry(0)->getIntrinsics(full_intrinsics);

      cv::Mat cameraMatrix(3,3,cv::DataType<double>::type);
      cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
      cameraMatrix.at<double>(0,0) = full_intrinsics[0];
      cameraMatrix.at<double>(1,1) = full_intrinsics[1];
      cameraMatrix.at<double>(0,2) = full_intrinsics[2];
      cameraMatrix.at<double>(1,2) = full_intrinsics[3];

      distCoeffs.at<double>(0) = full_intrinsics[4];
      distCoeffs.at<double>(1) = full_intrinsics[5];
      distCoeffs.at<double>(2) = full_intrinsics[6];
      distCoeffs.at<double>(3) = full_intrinsics[7];

      //compute the 3d position of the new keyframe relative to the matched frame
      //we will use ransac to determine which points are inliers
      cv::Mat rvec(3,1,cv::DataType<double>::type);
      cv::Mat tvec(3,1,cv::DataType<double>::type);

      std::vector<int> pnpInliers; 

      bool pnp_success = false;
      if(newFeaturesFinal.size()>30){
        pnp_success = solvePnPRansac(matchedPoints3d, newFeaturesFinal, 
          cameraMatrix, distCoeffs, rvec, tvec, false, 100, 2.0,
          0.5,pnpInliers);
      }

      //convert to eigen transform
      cv::Mat R; 
      cv::Rodrigues(rvec,R);
      tvec= -R*tvec;

      Eigen::Matrix3d rot;
      cv::cv2eigen(R,rot);

      Eigen::Vector3d trans;
      cv::cv2eigen(tvec,trans);
      Eigen::Affine3d transform_estimate = Eigen::Translation3d(trans)*Eigen::Quaterniond(rot);

      //check if transform is valid, meaning we were able to compute
      //a transform that enough points agreed upon
      if(pnp_success && pnpInliers.size()/(float)newFeaturesFinal.size()>0.301){
        //.301 was experimentally determined to give good results
        //it is the % required inliers for the loop closure to be good

        loopClosure=true;

        //if succesful add a constaint to the pose graph
        okvis::kinematics::Transformation okvis_estimate(transform_estimate.matrix());
        Pose3dNode newConstraintNode;
        newConstraintNode.q = okvis_estimate.q();
        newConstraintNode.p = okvis_estimate.r();
        Constraint3dNode newConstraint;
        newConstraint.t_be = newConstraintNode;
        newConstraint.id_begin = qret[0].Id;
        newConstraint.id_end = poses_.size();
        newConstraint.information = Eigen::Matrix<double,6,6>::Identity()/4;

        constraints_.push_back(newConstraint);

        //perform optimization of pose graph. 
        ::ceres::Problem problem;
        BuildOptimizationProblem(&problem);
        SolveOptimizationProblem(&problem);
        path_out.clear();
        for (std::map<int, Pose3dNode, std::less<int>,
                      Eigen::aligned_allocator<std::pair<const int, Pose3dNode> > >::
                 const_iterator poses_iter = nodes_.begin();
             poses_iter != nodes_.end(); ++poses_iter) {
          const std::map<int, Pose3dNode, std::less<int>,
                         Eigen::aligned_allocator<std::pair<const int, Pose3dNode> > >::
              value_type& pair = *poses_iter;
          path_out.push_back(okvis::kinematics::Transformation(pair.second.p,pair.second.q));    
        }

        //update current position and position of new keyframe. 
        currentKeyframeT_WSo= okvis::kinematics::Transformation(nodes_[poses_.size()].p,nodes_[poses_.size()].q);
        //kf->T_WS = currentKeyframeT_WSo;

        //std::cout << "LOOP CLOSURE: " << pnpInliers.size()/(float)newFeaturesFinal.size() << "%% inliers" << std::endl;
        //posesSinceLastLoop_=0;
      }
    }

    //add keyframe to graph
    lastEntry_ = kf->id = db_->add(kf->bowVec);
    poses_.push_back(kf);
    posesSinceLastLoop_++;
  }


  std::unique_ptr<OrbVocabulary> vocab_;
  std::unique_ptr<OrbDatabase> db_;
  std::vector<KeyFrameData::Ptr> poses_;

  okvis::kinematics::Transformation lastKeyframeT_SoW;
  okvis::kinematics::Transformation currentKeyframeT_WSo;

  size_t posesSinceLastLoop_;
  DBoW2::EntryId lastEntry_;

  VectorOfConstraints constraints_;
  VectorOfGravityConstraints gravity_;
  MapOfPoses nodes_;
  MapOfPoses originalNodes_;
  okvis::VioParameters parameters_;

private: 

 };


} //namespace okvis


#endif //INCLUDE_OKVIS_POSE_GRAPH_HPP_