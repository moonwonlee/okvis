/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Aug 21, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file ThreadedKFVio.cpp
 * @brief Source file for the ThreadedKFVio class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <map>

#include <glog/logging.h>

#include <okvis/ThreadedKFVio.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

static const int max_camera_input_queue_size = 10;
static const okvis::Duration temporal_imu_data_overlap(0.02);  // overlap of imu data before and after two consecutive frames [seconds]

#ifdef USE_MOCK
// Constructor for gmock.
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters, okvis::MockVioBackendInterface& estimator,
    okvis::MockVioFrontendInterface& frontend)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      estimator_(estimator),
      frontend_(frontend),
      parameters_(parameters),
      maxImuInputQueueSize_(60),
      keyframeSet_(false),
      poseGraph_("ORBvoc.yml") {
  init();
}
#else
// Constructor.
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      estimator_(),
      frontend_(parameters.nCameraSystem.numCameras()),
      parameters_(parameters),
      maxImuInputQueueSize_(
          2 * max_camera_input_queue_size * parameters.imu.rate
              / parameters.sensors_information.cameraRate), 
      keyframeSet_(false),
      poseGraph_("ORBvoc.yml") {
  setBlocking(false);
  init();
}
#endif

// Initialises settings and calls startThreads().
void ThreadedKFVio::init() {
  assert(parameters_.nCameraSystem.numCameras() > 0);
  numCameras_ = parameters_.nCameraSystem.numCameras();
  numCameraPairs_ = 1;

  frontend_.setBriskDetectionOctaves(parameters_.optimization.detectionOctaves);
  frontend_.setBriskDetectionThreshold(parameters_.optimization.detectionThreshold);
  frontend_.setBriskDetectionMaximumKeypoints(parameters_.optimization.maxNoKeypoints);

  lastOptimizedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)
  lastAddedStateTimestamp_ = okvis::Time(0.0) + temporal_imu_data_overlap;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)

  estimator_.addImu(parameters_.imu);
  for (size_t i = 0; i < numCameras_; ++i) {
    // parameters_.camera_extrinsics is never set (default 0's)...
    // do they ever change?
    estimator_.addCamera(parameters_.camera_extrinsics);
    cameraMeasurementsReceived_.emplace_back(
          std::shared_ptr<threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > >
          (new threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> >()));
  }
  
  // set up windows so things don't crash on Mac OS
  if(parameters_.visualization.displayImages){
    for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
      std::stringstream windowname;
      windowname << "OKVIS camera " << im;
  	  cv::namedWindow(windowname.str());
    }
  }
  
  startThreads();
}

// Start all threads.
void ThreadedKFVio::startThreads() {

  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.emplace_back(&ThreadedKFVio::frameConsumerLoop, this, i);
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.emplace_back(&ThreadedKFVio::matchingLoop, this);
  }
  imuConsumerThread_ = std::thread(&ThreadedKFVio::imuConsumerLoop, this);
  positionConsumerThread_ = std::thread(&ThreadedKFVio::positionConsumerLoop,
                                        this);
  gpsConsumerThread_ = std::thread(&ThreadedKFVio::gpsConsumerLoop, this);
  magnetometerConsumerThread_ = std::thread(
      &ThreadedKFVio::magnetometerConsumerLoop, this);
  differentialConsumerThread_ = std::thread(
      &ThreadedKFVio::differentialConsumerLoop, this);

  // algorithm threads
  visualizationThread_ = std::thread(&ThreadedKFVio::visualizationLoop, this);
  optimizationThread_ = std::thread(&ThreadedKFVio::optimizationLoop, this);
  keyframeProcessorThread_ = std::thread(&ThreadedKFVio::keyframeProcessorLoop, this);
  publisherThread_ = std::thread(&ThreadedKFVio::publisherLoop, this);
}

// Destructor. This calls Shutdown() for all threadsafe queues and joins all threads.
ThreadedKFVio::~ThreadedKFVio() {
  for (size_t i = 0; i < numCameras_; ++i) {
    cameraMeasurementsReceived_.at(i)->Shutdown();
  }
  keypointMeasurements_.Shutdown();
  matchedFrames_.Shutdown();
  imuMeasurementsReceived_.Shutdown();
  optimizationResults_.Shutdown();
  visualizationData_.Shutdown();
  imuFrameSynchronizer_.shutdown();
  positionMeasurementsReceived_.Shutdown();
  keyframeData_.Shutdown();

  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.at(i).join();
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.at(i).join();
  }
  imuConsumerThread_.join();
  positionConsumerThread_.join();
  gpsConsumerThread_.join();
  magnetometerConsumerThread_.join();
  differentialConsumerThread_.join();
  visualizationThread_.join();
  optimizationThread_.join();
  keyframeProcessorThread_.join();
  publisherThread_.join();

  /*okvis::kinematics::Transformation endPosition;
  estimator_.get_T_WS(estimator_.currentFrameId(), endPosition);
  std::stringstream s;
  s << endPosition.r();
  LOG(INFO) << "Sensor end position:\n" << s.str();
  LOG(INFO) << "Distance to origin: " << endPosition.r().norm();*/
#ifndef DEACTIVATE_TIMERS
  LOG(INFO) << okvis::timing::Timing::print();
#endif
}

// Add a new image.
bool ThreadedKFVio::addImage(const okvis::Time & stamp, size_t cameraIndex,
                             const cv::Mat & image,
                             const std::vector<cv::KeyPoint> * keypoints,
                             bool* /*asKeyframe*/) {
  assert(cameraIndex<numCameras_);

  if (lastAddedImageTimestamp_ > stamp
      && fabs((lastAddedImageTimestamp_ - stamp).toSec())
          > parameters_.sensors_information.frameTimestampTolerance) {
    LOG(ERROR)
        << "Received image from the past. Dropping the image.";
    LOG(ERROR)
        << "Last: " << lastAddedImageTimestamp_ << " Current: " << stamp;
    return false;
  }
  lastAddedImageTimestamp_ = stamp;

  std::shared_ptr<okvis::CameraMeasurement> frame = std::make_shared<
      okvis::CameraMeasurement>();
  frame->measurement.image = image;
  frame->timeStamp = stamp;
  frame->sensorId = cameraIndex;

  if (keypoints != nullptr) {
    frame->measurement.deliversKeypoints = true;
    frame->measurement.keypoints = *keypoints;
  } else {
    frame->measurement.deliversKeypoints = false;
  }

  if (blocking_) {
    cameraMeasurementsReceived_[cameraIndex]->PushBlockingIfFull(frame, 1);
    return true;
  } else {
    cameraMeasurementsReceived_[cameraIndex]->PushNonBlockingDroppingIfFull(
        frame, max_camera_input_queue_size);
    return cameraMeasurementsReceived_[cameraIndex]->Size() == 1;
  }
}

// Add an abstracted image observation.
bool ThreadedKFVio::addKeypoints(
    const okvis::Time & /*stamp*/, size_t /*cameraIndex*/,
    const std::vector<cv::KeyPoint> & /*keypoints*/,
    const std::vector<uint64_t> & /*landmarkIds*/,
    const cv::Mat & /*descriptors*/,
    bool* /*asKeyframe*/) {
  OKVIS_THROW(
      Exception,
      "ThreadedKFVio::addKeypoints() not implemented anymore since changes to _keypointMeasurements queue.");
  return false;
}

// Add an IMU measurement.
bool ThreadedKFVio::addImuMeasurement(const okvis::Time & stamp,
                                      const Eigen::Vector3d & alpha,
                                      const Eigen::Vector3d & omega) {

  okvis::ImuMeasurement imu_measurement;
  imu_measurement.measurement.accelerometers = alpha;
  imu_measurement.measurement.gyroscopes = omega;
  imu_measurement.timeStamp = stamp;

  if (blocking_) {
    imuMeasurementsReceived_.PushBlockingIfFull(imu_measurement, 1);
    return true;
  } else {
    imuMeasurementsReceived_.PushNonBlockingDroppingIfFull(
        imu_measurement, maxImuInputQueueSize_);
    return imuMeasurementsReceived_.Size() == 1;
  }
}

// Add a position measurement.
void ThreadedKFVio::addPositionMeasurement(const okvis::Time & stamp,
                                           const Eigen::Vector3d & position,
                                           const Eigen::Vector3d & positionOffset,
                                           const Eigen::Matrix3d & positionCovariance) {
  okvis::PositionMeasurement position_measurement;
  position_measurement.measurement.position = position;
  position_measurement.measurement.positionOffset = positionOffset;
  position_measurement.measurement.positionCovariance = positionCovariance;
  position_measurement.timeStamp = stamp;

  if (blocking_) {
    positionMeasurementsReceived_.PushBlockingIfFull(position_measurement, 1);
    return;
  } else {
    positionMeasurementsReceived_.PushNonBlockingDroppingIfFull(
        position_measurement, maxPositionInputQueueSize_);
    return;
  }
}

// Add a GPS measurement.
void ThreadedKFVio::addGpsMeasurement(const okvis::Time &, double, double,
                                      double, const Eigen::Vector3d &,
                                      const Eigen::Matrix3d &) {
  OKVIS_THROW(Exception, "GPS measurements not supported")
}

// Add a magnetometer measurement.
void ThreadedKFVio::addMagnetometerMeasurement(const okvis::Time &,
                                               const Eigen::Vector3d &, double) {
  OKVIS_THROW(Exception, "Magnetometer measurements not supported")
}

// Add a static pressure measurement.
void ThreadedKFVio::addBarometerMeasurement(const okvis::Time &, double, double) {

  OKVIS_THROW(Exception, "Barometer measurements not supported")
}

// Add a differential pressure measurement.
void ThreadedKFVio::addDifferentialPressureMeasurement(const okvis::Time &,
                                                       double, double) {

  OKVIS_THROW(Exception, "Differential pressure measurements not supported")
}

// Set the blocking variable that indicates whether the addMeasurement() functions
// should return immediately (blocking=false), or only when the processing is complete.
void ThreadedKFVio::setBlocking(bool blocking) {
  blocking_ = blocking;
  // disable time limit for optimization
  if(blocking_) {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    estimator_.setOptimizationTimeLimit(-1.0,parameters_.optimization.max_iterations);
  }
}

// Loop to process frames from camera with index cameraIndex
void ThreadedKFVio::frameConsumerLoop(size_t cameraIndex) {
  std::shared_ptr<okvis::CameraMeasurement> frame;
  std::shared_ptr<okvis::MultiFrame> multiFrame;
  TimerSwitchable beforeDetectTimer("1.1 frameLoopBeforeDetect"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer("1.1.1 waitForFrameSynchronizerMutex"+std::to_string(cameraIndex),true);
  TimerSwitchable addNewFrameToSynchronizerTimer("1.1.2 addNewFrameToSynchronizer"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForStateVariablesMutexTimer("1.1.3 waitForStateVariablesMutex"+std::to_string(cameraIndex),true);
  TimerSwitchable propagationTimer("1.1.4 propagationTimer"+std::to_string(cameraIndex),true);
  TimerSwitchable detectTimer("1.2 detectAndDescribe"+std::to_string(cameraIndex),true);
  TimerSwitchable afterDetectTimer("1.3 afterDetect"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer2("1.3.1 waitForFrameSynchronizerMutex2"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForMatchingThreadTimer("1.4 waitForMatchingThread"+std::to_string(cameraIndex),true);


  for (;;) {
    // get data and check for termination request
    if (cameraMeasurementsReceived_[cameraIndex]->PopBlocking(&frame) == false) {
      return;
    }
    beforeDetectTimer.start();
    {  // lock the frame synchronizer
      waitForFrameSynchronizerMutexTimer.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer.stop();
      // add new frame to frame synchronizer and get the MultiFrame containing it
      addNewFrameToSynchronizerTimer.start();
      multiFrame = frameSynchronizer_.addNewFrame(frame);
      addNewFrameToSynchronizerTimer.stop();
    }  // unlock frameSynchronizer only now as we can be sure that not two states are added for the same timestamp
    okvis::kinematics::Transformation T_WS;
    okvis::Time lastTimestamp;
    okvis::SpeedAndBias speedAndBiases;
    // copy last state variables
    {
      waitForStateVariablesMutexTimer.start();
      std::lock_guard<std::mutex> lock(lastState_mutex_);
      waitForStateVariablesMutexTimer.stop();
      T_WS = lastOptimized_T_WS_;
      speedAndBiases = lastOptimizedSpeedAndBiases_;
      lastTimestamp = lastOptimizedStateTimestamp_;
    }

    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime = multiFrame->timestamp()
        + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastTimestamp - temporal_imu_data_overlap;

    OKVIS_ASSERT_TRUE_DBG(Exception,imuDataBeginTime < imuDataEndTime,"imu data end time is smaller than begin time.");

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(
      okvis::Time(imuDataEndTime)) == false)  {
      return;
    }
    OKVIS_ASSERT_TRUE_DBG(Exception,
                          imuDataEndTime < imuMeasurements_.back().timeStamp,
                          "Waiting for up to date imu data seems to have failed!");

    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,
                                                           imuDataEndTime);

    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0) {
      beforeDetectTimer.stop();
      continue;
    }

    if (imuData.front().timeStamp > frame->timeStamp) {
      LOG(WARNING) << "Frame is newer than oldest IMU measurement. Dropping it.";
      beforeDetectTimer.stop();
      continue;
    }

    // get T_WC(camIndx) for detectAndDescribe()
    if (estimator_.numFrames() == 0) {
      // first frame ever
      bool success = okvis::Estimator::initPoseFromImu(imuData, T_WS);
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        lastOptimized_T_WS_ = T_WS;
        lastOptimizedSpeedAndBiases_.setZero();
        lastOptimizedSpeedAndBiases_.segment<3>(6) = imu_params_.a0;
        lastOptimizedStateTimestamp_ = multiFrame->timestamp();
      }
      OKVIS_ASSERT_TRUE_DBG(Exception, success,
          "pose could not be initialized from imu measurements.");
      if (!success) {
        beforeDetectTimer.stop();
        continue;
      }
    } else {
      // get old T_WS
      propagationTimer.start();
      okvis::ceres::ImuError::propagation(imuData, parameters_.imu, T_WS,
                                          speedAndBiases, lastTimestamp,
                                          multiFrame->timestamp());
      propagationTimer.stop();
    }
    okvis::kinematics::Transformation T_WC = T_WS
        * (*parameters_.nCameraSystem.T_SC(frame->sensorId));
    beforeDetectTimer.stop();
    detectTimer.start();
    frontend_.detectAndDescribe(frame->sensorId, multiFrame, T_WC, nullptr);
    detectTimer.stop();
    afterDetectTimer.start();

    bool push = false;
    {  // we now tell frame synchronizer that detectAndDescribe is done for MF with our timestamp
      waitForFrameSynchronizerMutexTimer2.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer2.stop();
      frameSynchronizer_.detectionEndedForMultiFrame(multiFrame->id());

      if (frameSynchronizer_.detectionCompletedForAllCameras(
          multiFrame->id())) {
//        LOG(INFO) << "detection completed for multiframe with id "<< multi_frame->id();
        push = true;
      }
    }  // unlocking frame synchronizer
    afterDetectTimer.stop();
    if (push) {
      // use queue size 1 to propagate a congestion to the _cameraMeasurementsReceived queue
      // and check for termination request
      waitForMatchingThreadTimer.start();
      if (keypointMeasurements_.PushBlockingIfFull(multiFrame, 1) == false) {
        return;
      }
      waitForMatchingThreadTimer.stop();
    }
  }
}

// Loop that matches frames with existing frames.
void ThreadedKFVio::matchingLoop() {
  TimerSwitchable prepareToAddStateTimer("2.1 prepareToAddState",true);
  TimerSwitchable waitForOptimizationTimer("2.2 waitForOptimization",true);
  TimerSwitchable addStateTimer("2.3 addState",true);
  TimerSwitchable matchingTimer("2.4 matching",true);

  for (;;) {
    // get new frame
    std::shared_ptr<okvis::MultiFrame> frame;

    // get data and check for termination request
    if (keypointMeasurements_.PopBlocking(&frame) == false)
      return;

    prepareToAddStateTimer.start();
    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime = frame->timestamp() + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastAddedStateTimestamp_
        - temporal_imu_data_overlap;

    OKVIS_ASSERT_TRUE_DBG(Exception,imuDataBeginTime < imuDataEndTime,
        "imu data end time is smaller than begin time." <<
        "current frametimestamp " << frame->timestamp() << " (id: " << frame->id() <<
        "last timestamp         " << lastAddedStateTimestamp_ << " (id: " << estimator_.currentFrameId());

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(
        okvis::Time(imuDataEndTime)) == false)
      return; OKVIS_ASSERT_TRUE_DBG(Exception,
        imuDataEndTime < imuMeasurements_.back().timeStamp,
        "Waiting for up to date imu data seems to have failed!");

    okvis::ImuMeasurementDeque imuData = getImuMeasurments(imuDataBeginTime,
                                                           imuDataEndTime);

    prepareToAddStateTimer.stop();
    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0)
      continue;

    // make sure that optimization of last frame is over.
    // TODO If we didn't actually 'pop' the _matchedFrames queue until after optimization this would not be necessary
    {
      waitForOptimizationTimer.start();
      std::unique_lock<std::mutex> l(estimator_mutex_);
      while (!optimizationDone_)
        optimizationNotification_.wait(l);
      waitForOptimizationTimer.stop();
      addStateTimer.start();
      okvis::Time t0Matching = okvis::Time::now();
      bool asKeyframe = false;
      if (estimator_.addStates(frame, imuData, asKeyframe)) {
        lastAddedStateTimestamp_ = frame->timestamp();
        addStateTimer.stop();
      } else {
        LOG(ERROR) << "Failed to add state! will drop multiframe.";
        addStateTimer.stop();
        continue;
      }

      // -- matching keypoints, initialising landmarks etc.
      okvis::kinematics::Transformation T_WS;
      estimator_.get_T_WS(frame->id(), T_WS);
      matchingTimer.start();
      frontend_.dataAssociationAndInitialization(estimator_, T_WS, parameters_,
                                                 map_, frame, &asKeyframe);
      matchingTimer.stop();

      if (asKeyframe){
        //JOE: 
        //since we have a new keyframe want to add old one to pose graph
        //may want to change to push blocking
        //may need to include landmark observations or other information
        //may need to increase queue size  

        //need to add current landmark positions to frame
        if(keyframeSet_){
          PoseGraph::KeyframeData::Ptr keyframeDataPtr = PoseGraph::KeyframeData::Ptr(
              new PoseGraph::KeyframeData());
          keyframeDataPtr->keyFrames = estimator_.multiFrame(
              estimator_.currentKeyframeId());
          estimator_.get_T_WS(estimator_.currentKeyframeId(),
                              keyframeDataPtr->T_WS);

          //transform from world to camera coordinates
          okvis::kinematics::Transformation keyframeT_CW = parameters_.nCameraSystem
              .T_SC(0)->inverse() * keyframeDataPtr->T_WS.inverse();

          //transform from new keyframe to old keyframe        
          keyframeDataPtr->T_SoSn = poseGraph_.lastKeyframeT_SoW*keyframeDataPtr->T_WS;  

          //Get current landmark positions
          //keyframeDataPtr->observations.resize(keyframeDataPtr->keyFrames->numKeypoints());
          okvis::MapPoint landmark;
          okvis::Observation it;
          //okvis::ObservationVector::iterator it = keyframeDataPtr
          //  ->observations.begin();
          for (size_t k = 0; k < keyframeDataPtr->keyFrames->numKeypoints(0); ++k) {
            //OKVIS_ASSERT_TRUE_DBG(Exception,it != keyframeDataPtr->observations.end(),"Observation-vector not big enough");
            it.landmarkId = keyframeDataPtr->keyFrames->landmarkId(0, k);
            it.keypointIdx = k;
            if (estimator_.isLandmarkAdded(it.landmarkId)) {
              estimator_.getLandmark(it.landmarkId, landmark);
              //landmark in camera coords
              it.landmark_C = keyframeT_CW*landmark.point;
              //if (estimator_.isLandmarkInitialized(it.landmarkId))
              keyframeDataPtr->observations.push_back(it);
            }
          }
          okvis::kinematics::Transformation lastT_SW = keyframeDataPtr->T_WS.inverse();
          if(!keyframeData_.PushNonBlockingDroppingIfFull(keyframeDataPtr, 1))
            poseGraph_.lastKeyframeT_SoW = lastT_SW;
        } else{
          keyframeSet_=true;
        }
        estimator_.setKeyframe(frame->id(), asKeyframe);
      } 
      if(!blocking_) {
        double timeLimit = parameters_.optimization.timeLimitForMatchingAndOptimization
                           -(okvis::Time::now()-t0Matching).toSec();
        estimator_.setOptimizationTimeLimit(std::max<double>(0.0, timeLimit),
                                            parameters_.optimization.min_iterations);
      }
      optimizationDone_ = false;
    }  // unlock estimator_mutex_

    // use queue size 1 to propagate a congestion to the _matchedFrames queue
    if (matchedFrames_.PushBlockingIfFull(frame, 1) == false)
      return;
  }
}

// Loop to process IMU measurements.
void ThreadedKFVio::imuConsumerLoop() {
  okvis::ImuMeasurement data;
  TimerSwitchable processImuTimer("0 processImuMeasurements",true);
  for (;;) {
    // get data and check for termination request
    if (imuMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    processImuTimer.start();
    okvis::Time start;
    const okvis::Time* end;  // do not need to copy end timestamp
    {
      std::lock_guard<std::mutex> imuLock(imuMeasurements_mutex_);
      OKVIS_ASSERT_TRUE(Exception,
                        imuMeasurements_.empty()
                        || imuMeasurements_.back().timeStamp < data.timeStamp,
                        "IMU measurement from the past received");

      if (parameters_.publishing.publishImuPropagatedState) {
        if (!repropagationNeeded_ && imuMeasurements_.size() > 0) {
          start = imuMeasurements_.back().timeStamp;
        } else if (repropagationNeeded_) {
          std::lock_guard<std::mutex> lastStateLock(lastState_mutex_);
          start = lastOptimizedStateTimestamp_;
          T_WS_propagated_ = lastOptimized_T_WS_;
          speedAndBiases_propagated_ = lastOptimizedSpeedAndBiases_;
          repropagationNeeded_ = false;
        } else
          start = okvis::Time(0, 0);
        end = &data.timeStamp;
      }
      imuMeasurements_.push_back(data);
    }  // unlock _imuMeasurements_mutex

    // notify other threads that imu data with timeStamp is here.
    imuFrameSynchronizer_.gotImuData(data.timeStamp);

    if (parameters_.publishing.publishImuPropagatedState) {
      Eigen::Matrix<double, 15, 15> covariance;
      Eigen::Matrix<double, 15, 15> jacobian;

      frontend_.propagation(imuMeasurements_, imu_params_, T_WS_propagated_,
                            speedAndBiases_propagated_, start, *end, &covariance,
                            &jacobian);
      OptimizationResults result;
      result.stamp = *end;
      result.T_WS = T_WS_propagated_;
      result.speedAndBiases = speedAndBiases_propagated_;
      result.omega_S = imuMeasurements_.back().measurement.gyroscopes
          - speedAndBiases_propagated_.segment<3>(3);
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        result.vector_of_T_SCi.push_back(
            okvis::kinematics::Transformation(
                *parameters_.nCameraSystem.T_SC(i)));
      }
      result.onlyPublishLandmarks = false;
      optimizationResults_.PushNonBlockingDroppingIfFull(result,1);
    }
    processImuTimer.stop();
  }
}

// Loop to process position measurements.
void ThreadedKFVio::positionConsumerLoop() {
  okvis::PositionMeasurement data;
  for (;;) {
    // get data and check for termination request
    if (positionMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    // collect
    {
      std::lock_guard<std::mutex> positionLock(positionMeasurements_mutex_);
      positionMeasurements_.push_back(data);
    }
  }
}

// Loop to process GPS measurements.
void ThreadedKFVio::gpsConsumerLoop() {
}

// Loop to process magnetometer measurements.
void ThreadedKFVio::magnetometerConsumerLoop() {
}

// Loop to process differential pressure measurements.
void ThreadedKFVio::differentialConsumerLoop() {
}

// Loop that visualizes completed frames.
void ThreadedKFVio::visualizationLoop() {
  okvis::VioVisualizer visualizer_(parameters_);
  for (;;) {
    VioVisualizer::VisualizationData::Ptr new_data;
    if (visualizationData_.PopBlocking(&new_data) == false)
      return;
    //visualizer_.showDebugImages(new_data);
    std::vector<cv::Mat> out_images(parameters_.nCameraSystem.numCameras());
    for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
      out_images[i] = visualizer_.drawMatches(new_data, i);
    }
	displayImages_.PushNonBlockingDroppingIfFull(out_images,1);
  }
}

// trigger display (needed because OSX won't allow threaded display)
void ThreadedKFVio::display() {
  std::vector<cv::Mat> out_images;
  if (displayImages_.Size() == 0)
	return;
  if (displayImages_.PopBlocking(&out_images) == false)
    return;
  // draw
  for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
    std::stringstream windowname;
    windowname << "OKVIS camera " << im;
    cv::imshow(windowname.str(), out_images[im]);
  }
  cv::waitKey(1);
}

void ThreadedKFVio::debugDisplay() {
  cv::Mat debug_image;
  if (debugImages_.Size() == 0)
  return;
  if (debugImages_.PopBlocking(&debug_image) == false)
    return;
  // draw
  cv::imshow("Debug", debug_image);
  
  cv::waitKey(1);
}


// Get a subset of the recorded IMU measurements.
okvis::ImuMeasurementDeque ThreadedKFVio::getImuMeasurments(
    okvis::Time& imuDataBeginTime, okvis::Time& imuDataEndTime) {
  // sanity checks:
  // if end time is smaller than begin time, return empty queue.
  // if begin time is larger than newest imu time, return empty queue.
  if (imuDataEndTime < imuDataBeginTime
      || imuDataBeginTime > imuMeasurements_.back().timeStamp)
    return okvis::ImuMeasurementDeque();

  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  // get iterator to imu data before previous frame
  okvis::ImuMeasurementDeque::iterator first_imu_package = imuMeasurements_
      .begin();
  okvis::ImuMeasurementDeque::iterator last_imu_package =
      imuMeasurements_.end();
  // TODO go backwards through queue. Is probably faster.
  for (auto iter = imuMeasurements_.begin(); iter != imuMeasurements_.end();
      ++iter) {
    // move first_imu_package iterator back until iter->timeStamp is higher than requested begintime
    if (iter->timeStamp <= imuDataBeginTime)
      first_imu_package = iter;

    // set last_imu_package iterator as soon as we hit first timeStamp higher than requested endtime & break
    if (iter->timeStamp >= imuDataEndTime) {
      last_imu_package = iter;
      // since we want to include this last imu measurement in returned Deque we
      // increase last_imu_package iterator once.
      ++last_imu_package;
      break;
    }
  }

  // create copy of imu buffer
  return okvis::ImuMeasurementDeque(first_imu_package, last_imu_package);
}

// Remove IMU measurements from the internal buffer.
int ThreadedKFVio::deleteImuMeasurements(const okvis::Time& eraseUntil) {
  std::lock_guard<std::mutex> lock(imuMeasurements_mutex_);
  if (imuMeasurements_.front().timeStamp > eraseUntil)
    return 0;

  okvis::ImuMeasurementDeque::iterator eraseEnd;
  int removed = 0;
  for (auto it = imuMeasurements_.begin(); it != imuMeasurements_.end(); ++it) {
    eraseEnd = it;
    if (it->timeStamp >= eraseUntil)
      break;
    ++removed;
  }

  imuMeasurements_.erase(imuMeasurements_.begin(), eraseEnd);

  return removed;
}

// Loop that performs the optimization and marginalisation.
void ThreadedKFVio::optimizationLoop() {
  TimerSwitchable optimizationTimer("3.1 optimization",true);
  TimerSwitchable marginalizationTimer("3.2 marginalization",true);
  TimerSwitchable afterOptimizationTimer("3.3 afterOptimization",true);

  for (;;) {
    std::shared_ptr<okvis::MultiFrame> frame_pairs;
    VioVisualizer::VisualizationData::Ptr visualizationDataPtr;
    okvis::Time deleteImuMeasurementsUntil(0, 0);
    if (matchedFrames_.PopBlocking(&frame_pairs) == false)
      return;
    OptimizationResults result;
    {
      std::lock_guard<std::mutex> l(estimator_mutex_);
      optimizationTimer.start();
      //if(frontend_.isInitialized()){
        estimator_.optimize(parameters_.optimization.max_iterations, 2, false);
      //}
      /*if (estimator_.numFrames() > 0 && !frontend_.isInitialized()){
        // undo translation
        for(size_t n=0; n<estimator_.numFrames(); ++n){
          okvis::kinematics::Transformation T_WS_0;
          estimator_.get_T_WS(estimator_.frameIdByAge(n),T_WS_0);
          Eigen::Matrix4d T_WS_0_mat = T_WS_0.T();
          T_WS_0_mat.topRightCorner<3,1>().setZero();
          estimator_.set_T_WS(estimator_.frameIdByAge(n),okvis::kinematics::Transformation(T_WS_0_mat));
          okvis::SpeedAndBias sb_0 = okvis::SpeedAndBias::Zero();
          if(estimator_.getSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0)){
            sb_0.head<3>().setZero();
            estimator_.setSpeedAndBias(estimator_.frameIdByAge(n), 0, sb_0);
          }
        }
      }*/

      optimizationTimer.stop();

      // get timestamp of last frame in IMU window. Need to do this before marginalization as it will be removed there (if not keyframe)
      if (estimator_.numFrames()
          > size_t(parameters_.optimization.numImuFrames)) {
        deleteImuMeasurementsUntil = estimator_.multiFrame(
            estimator_.frameIdByAge(parameters_.optimization.numImuFrames))
            ->timestamp() - temporal_imu_data_overlap;
      }

      marginalizationTimer.start();
      estimator_.applyMarginalizationStrategy(
          parameters_.optimization.numKeyframes,
          parameters_.optimization.numImuFrames, result.transferredLandmarks);
      marginalizationTimer.stop();
      afterOptimizationTimer.start();

      // now actually remove measurements
      deleteImuMeasurements(deleteImuMeasurementsUntil);

      // saving optimized state and saving it in OptimizationResults struct
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        estimator_.get_T_WS(frame_pairs->id(), lastOptimized_T_WS_);
        estimator_.getSpeedAndBias(frame_pairs->id(), 0,
                                   lastOptimizedSpeedAndBiases_);
        lastOptimizedStateTimestamp_ = frame_pairs->timestamp();

        // if we publish the state after each IMU propagation we do not need to publish it here.
        if (!parameters_.publishing.publishImuPropagatedState) {
          result.T_WS = lastOptimized_T_WS_;
          result.speedAndBiases = lastOptimizedSpeedAndBiases_;
          result.stamp = lastOptimizedStateTimestamp_;
          result.onlyPublishLandmarks = false;
        }
        else
          result.onlyPublishLandmarks = true;
        estimator_.getLandmarks(result.landmarksVector);

        repropagationNeeded_ = true;
      }

      if (parameters_.visualization.displayImages) {
        // fill in information that requires access to estimator.
        visualizationDataPtr = VioVisualizer::VisualizationData::Ptr(
            new VioVisualizer::VisualizationData());
        visualizationDataPtr->observations.resize(frame_pairs->numKeypoints());
        okvis::MapPoint landmark;
        okvis::ObservationVector::iterator it = visualizationDataPtr
            ->observations.begin();
        for (size_t camIndex = 0; camIndex < frame_pairs->numFrames();
            ++camIndex) {
          for (size_t k = 0; k < frame_pairs->numKeypoints(camIndex); ++k) {
            OKVIS_ASSERT_TRUE_DBG(Exception,it != visualizationDataPtr->observations.end(),"Observation-vector not big enough");
            it->keypointIdx = k;
            frame_pairs->getKeypoint(camIndex, k, it->keypointMeasurement);
            frame_pairs->getKeypointSize(camIndex, k, it->keypointSize);
            it->cameraIdx = camIndex;
            it->frameId = frame_pairs->id();
            it->landmarkId = frame_pairs->landmarkId(camIndex, k);
            if (estimator_.isLandmarkAdded(it->landmarkId)) {
              estimator_.getLandmark(it->landmarkId, landmark);
              it->landmark_W = landmark.point;
              if (estimator_.isLandmarkInitialized(it->landmarkId))
                it->isInitialized = true;
              else
                it->isInitialized = false;
            } else {
              it->landmark_W = Eigen::Vector4d(0, 0, 0, 0);  // set to infinity to tell visualizer that landmark is not added
            }
            ++it;
          }
        }
        visualizationDataPtr->keyFrames = estimator_.multiFrame(
            estimator_.currentKeyframeId());
        estimator_.get_T_WS(estimator_.currentKeyframeId(),
                            visualizationDataPtr->T_WS_keyFrame);
      }

      optimizationDone_ = true;
    }  // unlock mutex
    optimizationNotification_.notify_all();

    if (!parameters_.publishing.publishImuPropagatedState) {
      // adding further elements to result that do not access estimator.
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        result.vector_of_T_SCi.push_back(
            okvis::kinematics::Transformation(
                *parameters_.nCameraSystem.T_SC(i)));
      }
    }
    optimizationResults_.Push(result);

    // adding further elements to visualization data that do not access estimator
    if (parameters_.visualization.displayImages) {
      visualizationDataPtr->currentFrames = frame_pairs;
      visualizationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr, 1);
    }
    afterOptimizationTimer.stop();
  }
}

// Loop that publishes the keyframes after leaving the optimization window
// Will probably change this to do the actual pose graph processing at some point
void ThreadedKFVio::keyframeProcessorLoop() {
  for(;;) {
    //Wait for new keyframe
    PoseGraph::KeyframeData::Ptr newKeyframe;
    if(keyframeData_.PopBlocking(&newKeyframe) == false)
      return;

    //All of the below should be moved to the pose graph.hpp file
    //compute most optimized location of the current keyframe
    poseGraph_.currentKeyframeT_WSo = poseGraph_.currentKeyframeT_WSo*newKeyframe->T_SoSn;

    //This is only for debugging purposes
    Pose3dNode node;
    node.q = newKeyframe->T_WS.q();
    node.p = newKeyframe->T_WS.r();
    poseGraph_.originalNodes_[poseGraph_.poses_.size()]=node;
    
    //Update the new keyframe pose using thew best estimate from the optimized pose graph
    newKeyframe->T_WS = poseGraph_.currentKeyframeT_WSo;
    node.q = newKeyframe->T_WS.q();
    node.p = newKeyframe->T_WS.r();
    //Add this node to the pose graph
    poseGraph_.nodes_[poseGraph_.poses_.size()]=node;

    //As long as this is not the first pose
    if(poseGraph_.poses_.size()>0){
      //add constraint to pose graph between this pose and previous pose
      Pose3dNode newDiffConstraintNode;
      newDiffConstraintNode.q = newKeyframe->T_SoSn.q();
      newDiffConstraintNode.p = newKeyframe->T_SoSn.r();

      Constraint3dNode newDiffConstraint;
      newDiffConstraint.t_be = newDiffConstraintNode;
      newDiffConstraint.id_begin = poseGraph_.poses_.size()-1;
      newDiffConstraint.id_end = poseGraph_.poses_.size();
      newDiffConstraint.information = Eigen::Matrix<double,6,6>::Identity();
      poseGraph_.constraints_.push_back(newDiffConstraint);

      //I also include a "gravity constraint" 
      //This represents the fact that the tilt and roll estimates are bounded due to 
      //the presence of an accelerometer
      GravityNode newGravityConstraint;
      newGravityConstraint.g = newKeyframe->T_WS.q()*Eigen::Vector3d(0,0,1);
      newGravityConstraint.id = poseGraph_.poses_.size();
      newGravityConstraint.information = Eigen::Matrix3d::Identity();
      poseGraph_.gravity_.push_back(newGravityConstraint);

    }

    //These are for debugging purposes
    poseGraph_.OutputPoses("poses.txt");
    poseGraph_.OutputPoses("orig_poses.txt", poseGraph_.originalNodes_);

    //This publishes the current pose for display purposes. Need to replace with full path publisher
    if (stateCallback_)
      stateCallback_(okvis::Time(),poseGraph_.currentKeyframeT_WSo);

    //Get feature points  
    std::vector<cv::KeyPoint> points;
    points.reserve(newKeyframe->observations.size());
    for (size_t k = 0; k < newKeyframe->observations.size(); ++k) {
      cv::KeyPoint kp;
      newKeyframe->keyFrames->getCvKeypoint(0,newKeyframe->observations[k].keypointIdx,kp);

      if(!okvis::in_image(kp.pt,newKeyframe->keyFrames->image(0))){
        std::cout << "Point Not in Image\n";
      }
      points.emplace_back(kp);
    }

    //detect orb descriptors
    //later will just use brisk descriptors, but need to create proper DBoW2 templates first
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->compute(newKeyframe->keyFrames->image(0),points,newKeyframe->descriptors);

    //if we can not compute orb descriptors for all points, do not use keyframe
    //this will not be an issue once we switch to brisk descriptors
    if(points.size()!=newKeyframe->observations.size())
      continue;

    //convert to DBoW descriptor format
    std::vector<cv::Mat> bowDesc;
    bowDesc.reserve(newKeyframe->descriptors.rows);
    for(int i=0; i<newKeyframe->descriptors.rows; i++){
      bowDesc.emplace_back(newKeyframe->descriptors.row(i));
    } 

    //get BoW vector for keyframe 
    poseGraph_.vocab_->transform(bowDesc, newKeyframe->bowVec);
    if(newKeyframe->bowVec.size()==0)
      continue;
    if(poseGraph_.posesSinceLastLoop_<20){
      //20 is the min number of keyframes before we want to consider a loop closure
      //this prevents us from trying to close the loop wuth very nearby frames
      poseGraph_.lastEntry_ = newKeyframe->id = poseGraph_.db_->add(newKeyframe->bowVec);
      poseGraph_.poses_.push_back(newKeyframe);
      poseGraph_.posesSinceLastLoop_++;
      continue;
    }

    DBoW2::QueryResults qret;
    //get best result from database. Don't consider the 20 most recent images
    //should experiment with getting top 3 or top 5 results from database
    poseGraph_.db_->query(newKeyframe->bowVec,qret,1,poseGraph_.lastEntry_-20);
    //compute similarity score to previous image
    //this is necessary as the score will depend on the number and type of features in the image
    //see DBoW2 paper for more details
    float baseScore = poseGraph_.vocab_->score(newKeyframe->bowVec,poseGraph_.poses_[poseGraph_.lastEntry_]->bowVec);
    if(qret.size()>0 && baseScore>0.1 && qret[0].Score/baseScore>0.75){ 
      //.01 is min similarity to previous frame,
      //.75 is required similarity of matched frame relative to previous frame
      //that is, the matched frame must be at least 75% as similar to the current
      //frame as the current frame is to the previous frame

      //if the matched frame is similar enough, attempt to compute SE3 transform
      PoseGraph::KeyframeData::Ptr matchedFrame = poseGraph_.poses_[qret[0].Id];

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
        matchedFrame->keyFrames->image(0), newKeyframe->keyFrames->image(0),
        matchedFeatures, newFeatures,valid,err,cv::Size(11,11),3,
        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
        cv::OPTFLOW_LK_GET_MIN_EIGENVALS, 1e-4);

      //again only for debugging
      //creates image showing matches between the two frames
      cv::Mat outImg(matchedFrame->keyFrames->image(0).rows,
        matchedFrame->keyFrames->image(0).cols+newKeyframe->keyFrames->image(0).cols, CV_8UC1);
      cv::Mat left(outImg, cv::Rect(0,0,matchedFrame->keyFrames->image(0).cols,matchedFrame->keyFrames->image(0).rows));
      cv::Mat right(outImg, cv::Rect(matchedFrame->keyFrames->image(0).cols,0,newKeyframe->keyFrames->image(0).cols,matchedFrame->keyFrames->image(0).rows));
      matchedFrame->keyFrames->image(0).copyTo(left);
      newKeyframe->keyFrames->image(0).copyTo(right);
      for(int i=0; i <matchedFeatures.size(); i++){
        if(valid[i])
          line(outImg, matchedFeatures[i],cv::Point2f(newFeatures[i].x+matchedFrame->keyFrames->image(0).cols,newFeatures[i].y),cv::Scalar(255,0,0));
      }
      debugImages_.PushNonBlockingDroppingIfFull(outImg, 1);

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
      newKeyframe->keyFrames->geometry(0)->getIntrinsics(full_intrinsics);

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
      solvePnPRansac(matchedPoints3d, newFeaturesFinal, 
        cameraMatrix, distCoeffs, rvec, tvec, false, 100, 2.0,
        0.5,pnpInliers);

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
      if(pnpInliers.size()/(float)newFeaturesFinal.size()>0.301){
        //.301 was experimentally determined to give good results
        //it is the % required inliers for the loop closure to be good

        //if succesful add a constaint to the pose graph
        okvis::kinematics::Transformation okvis_estimate(transform_estimate.matrix());
        Pose3dNode newConstraintNode;
        newConstraintNode.q = okvis_estimate.q();
        newConstraintNode.p = okvis_estimate.r();
        Constraint3dNode newConstraint;
        newConstraint.t_be = newConstraintNode;
        newConstraint.id_begin = qret[0].Id;
        newConstraint.id_end = poseGraph_.poses_.size();
        newConstraint.information = Eigen::Matrix<double,6,6>::Identity()/4;

        poseGraph_.constraints_.push_back(newConstraint);

        //perform optimization of pose graph. 
        ::ceres::Problem problem;
        poseGraph_.BuildOptimizationProblem(&problem);
        poseGraph_.SolveOptimizationProblem(&problem);
        poseGraph_.OutputPoses("poses.txt");

        //update current position and position of new keyframe. 
        poseGraph_.currentKeyframeT_WSo= okvis::kinematics::Transformation(poseGraph_.nodes_[poseGraph_.poses_.size()].p,poseGraph_.nodes_[poseGraph_.poses_.size()].q);
        newKeyframe->T_WS = poseGraph_.currentKeyframeT_WSo;

        //std::cout << "LOOP CLOSURE: " << pnpInliers.size()/(float)newFeaturesFinal.size() << "%% inliers" << std::endl;
        //poseGraph_.posesSinceLastLoop_=0;
      }
    }

    //add keyframe to graph
    poseGraph_.lastEntry_ = newKeyframe->id = poseGraph_.db_->add(newKeyframe->bowVec);
    poseGraph_.poses_.push_back(newKeyframe);
    poseGraph_.posesSinceLastLoop_++;
  }
}

// Loop that publishes the newest state and landmarks.
void ThreadedKFVio::publisherLoop() {
  for (;;) {
    // get the result data
    OptimizationResults result;
    if (optimizationResults_.PopBlocking(&result) == false)
      return;

    // call all user callbacks
    //if (stateCallback_ && !result.onlyPublishLandmarks)
      //stateCallback_(result.stamp, result.T_WS);
    if (fullStateCallback_ && !result.onlyPublishLandmarks)
      fullStateCallback_(result.stamp, result.T_WS, result.speedAndBiases,
                         result.omega_S);
    if (fullStateCallbackWithExtrinsics_ && !result.onlyPublishLandmarks)
      fullStateCallbackWithExtrinsics_(result.stamp, result.T_WS,
                                       result.speedAndBiases, result.omega_S,
                                       result.vector_of_T_SCi);
    if (landmarksCallback_ && !result.landmarksVector.empty())
      landmarksCallback_(result.stamp, result.landmarksVector,
                         result.transferredLandmarks);  //TODO(gohlp): why two maps?
  }
}

}  // namespace okvis
