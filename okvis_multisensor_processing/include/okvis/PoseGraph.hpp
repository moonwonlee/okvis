#ifndef INCLUDE_OKVIS_POSE_GRAPH_HPP_
#define INCLUDE_OKVIS_POSE_GRAPH_HPP_

#include <okvis/assert_macros.hpp>

#include <okvis/Parameters.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/FrameTypedefs.hpp>


#include <DBoW2.h>

/// \brief okvis Main namespace of this package.
namespace okvis {

/**
 * @brief This class is responsible to visualize the matching results
 */
class PoseGraph {
 public:

  PoseGraph(std::string vocabPath):
  posesSinceLastLoop_(0){
      std::cout << "Loading Vocabulary From: " << vocabPath << std::endl;
      vocab_.reset(new OrbVocabulary(vocabPath));
      std::cout << "Vocabulary Size: " << vocab_->size() << std::endl;
      db_.reset(new OrbDatabase(*vocab_));
  }

  /// @brief This struct contains the relevant data for visualizing
  struct KeyframeData {
    typedef std::shared_ptr<KeyframeData> Ptr;
    std::shared_ptr<okvis::MultiFrame> keyFrames;     ///< Current keyframe.
    okvis::kinematics::Transformation T_WS_keyFrame;  ///< Pose of the current keyframe
    okvis::ObservationVector observations;
    DBoW2::EntryId id;
    DBoW2::BowVector bowVec;
  };

  std::unique_ptr<OrbVocabulary> vocab_;
  std::unique_ptr<OrbDatabase> db_;
  std::vector<KeyframeData::Ptr> poses_;

  size_t posesSinceLastLoop_;
  DBoW2::EntryId lastEntry_;

private: 

 };


} //namespace okvis


#endif //INCLUDE_OKVIS_POSE_GRAPH_HPP_