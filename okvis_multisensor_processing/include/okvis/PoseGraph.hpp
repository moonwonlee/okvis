#ifndef INCLUDE_OKVIS_POSE_GRAPH_HPP_
#define INCLUDE_OKVIS_POSE_GRAPH_HPP_

#include <okvis/assert_macros.hpp>

#include <okvis/Parameters.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/FrameTypedefs.hpp>
#include "point_cloud.hpp"
#include "correspondence_ransac.hpp"

#define RANSAC_POINTS 3 //Number of random samples per iteration
#define RANSAC_THRESHOLD .2f //Allowable distance in m to be inlier for object 1m away
#define RANSAC_ITERATIONS 100 //Number of ransac iterations to perform



#include <DBoW2.h>

/// \brief okvis Main namespace of this package.
namespace okvis {

inline bool in_image(cv::Point pt, const cv::Mat& image){
  return pt.y>0 && pt.y < image.rows && pt.x>0 && pt.x < image.cols;
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

private: 

 };


} //namespace okvis


#endif //INCLUDE_OKVIS_POSE_GRAPH_HPP_