#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  ///* State dimension
  n_x_ = 5;

  ///* Augmented state dimension
  n_aug_ = 7;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.4 * M_PI;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  // complete the initialization.


  ///* initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;
  ///* predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  ///* time when the state is true, in us
  time_us_ = 0;

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  ///* Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);

  // Set weights
  for (int i = 0; i < (2 * n_aug_ + 1); i++)
  {
    //set weights
    if (i != 0)
    {
      weights_(i) = 1 / (2 * (lambda_ + n_aug_));
    }
    else
    {
      weights_(i) = lambda_ / (lambda_ + n_aug_);
    }
  }
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**

  Make sure you switch between lidar and radar measurements.
  */

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    float px, py; //velocity is not available from first measurement

    // first measurement

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      px = meas_package.raw_measurements_[0]*cos(meas_package.raw_measurements_[1]);
      py = meas_package.raw_measurements_[0]*sin(meas_package.raw_measurements_[1]);

    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
    }

    x_ << px, py, 0, 0, 0;  //velocity is not available from first measurement
    time_us_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Update the state transition matrix F according to the new elapsed time.
   - Time is measured in seconds. */
  //Compute the time elapsed between the current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // Perform prediction step
  Prediction(dt);

  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR)
      && (true == use_radar_)) {
    UpdateRadar(meas_package);
  }

  if ((meas_package.sensor_type_ == MeasurementPackage::LASER)
      && (true == use_laser_)) {
    UpdateLidar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  /**

  Estimate the object's location. Modify the state vector, x_.
  Predict sigma points, the state, and the state covariance matrix.
  */

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug << x_, 0, 0;

  //create augmented covariance matrix
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_*std_a_, 0,
       0,std_yawdd_*std_yawdd_;

  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug.bottomRightCorner(2,2) = Q;

  //create square root matrix
  //calculate square root of P
  MatrixXd A = P_aug.llt().matrixL();

  //create augmented sigma points
  MatrixXd tmp = sqrt(lambda_ + n_aug_) * A;

  MatrixXd B = tmp.colwise() + x_aug;
  MatrixXd C = (-tmp).colwise() + x_aug;

  Xsig_aug << x_aug, B, C;

  double x0, x1, x3;

  double u0, u1, u2, u3, u4;

  //Reset state vector
  x_.fill(0.0);

  for (int i = 0; i < (2 * n_aug_ + 1); i++)
  {
    //predict sigma points
    double v = Xsig_aug(2,i);
    double psi = Xsig_aug(3,i);
    double psi_dot = Xsig_aug(4,i);
    double ua = Xsig_aug(5,i);
    double upsi_dd = Xsig_aug(6,i);

    //avoid division by zero
    if (fabs(psi_dot) > 0.001) {
      x0 = (v / psi_dot ) * (sin(psi + psi_dot* delta_t) - sin(psi));
      x1 = (v / psi_dot ) * (-cos(psi + psi_dot* delta_t) + cos(psi));
      x3 = psi_dot * delta_t;

    } else {
      x0 = v * cos(psi) * delta_t;
      x1 = v * sin(psi) * delta_t;
      x3 = 0;
    }

    u0 = 0.5 * delta_t * delta_t * cos(psi) * ua;
    u1 = 0.5 * delta_t * delta_t * sin(psi) * ua;
    u2 = delta_t * ua;
    u3 = 0.5 * delta_t * delta_t * upsi_dd;
    u4 = delta_t * upsi_dd;

    //write predicted sigma points into right column
    VectorXd x_int = VectorXd(5);
    x_int << x0, x1, 0, x3, 0;
    VectorXd u = VectorXd(5);
    u << u0, u1, u2, u3, u4;

    Xsig_pred_.col(i) = Xsig_aug.col(i).head(5) + x_int + u;

    //predict state mean
    x_ += (weights_(i) *  Xsig_pred_.col(i));
  }

  //Reset state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < (2 * n_aug_ + 1); i++)
  {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff[3] = atan2(sin(x_diff[3]),cos(x_diff[3])); // normalize phi

    //predict state covariance matrix
    P_ += (weights_(i) * x_diff * (x_diff.transpose()));
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**

  Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  */

  //set measurement dimension, Lidar can measure px and py;
  int n_z = 2;
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //Vector of incoming radar measurement
  VectorXd z = VectorXd(n_z); //px and py in m
  z << meas_package.raw_measurements_[0],
       meas_package.raw_measurements_[1];

  //Reset predicted measurement mean vector
  z_pred.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //recover state parameters
    Zsig(0,i) = Xsig_pred_(0,i);
    Zsig(1,i) = Xsig_pred_(1,i);

    //predict measurement mean
    z_pred += (weights_(i) *  Zsig.col(i));

  }

  //Reset measurement covariance matrix S
  S.fill(0.0);
  //Reset cross correlation matrix
  Tc.fill(0.0);

  //calculate measurement covariance matrix S
  for (int i = 0; i < (2 * n_aug_ + 1); i++)
  {
    // state difference
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //calculate measurement covariance matrix
    S += (weights_(i) * z_diff * (z_diff.transpose()));
    //calculate cross correlatione matrix
    Tc += (weights_(i) * x_diff * (z_diff.transpose()));
  }

  //measurement noise matrix R
  MatrixXd R = MatrixXd(n_z,n_z);
  R << std_laspx_*std_laspx_,0,
      0,std_laspy_*std_laspy_;

  // Add measurement noise R
  S += R;

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ += (K * z_diff);
  P_ -= (K * S * K.transpose());
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**

  Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  */

  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //Vector of incoming radar measurement
  VectorXd z = VectorXd(n_z); //rho in m, phi in rad, rho_dot in m/s
  z << meas_package.raw_measurements_[0],
       meas_package.raw_measurements_[1],
       meas_package.raw_measurements_[2];

  //Reset mean predicted measurement
  z_pred.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    //recover state parameters
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    //transform sigma points into measurement space
    Zsig(0,i) = sqrt(pow(px, 2) + pow(py, 2));
    Zsig(1,i) = atan2(py, px);  //Used atan2 to elimnate the need of phi range check
    Zsig(2, i) = (px * v * cos(yaw) + py * v * sin(yaw)) / Zsig(0, i);

    //predict measurement mean
     z_pred += (weights_(i) *  Zsig.col(i));
  }

  //Reset measurement covariance matrix S
  S.fill(0.0);
  //Reset cross correlation matrix
  Tc.fill(0.0);
  //Calculate measurement covariance matrix S
  for (int i = 0; i < (2 * n_aug_ + 1); i++)
  {
    // state difference
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    x_diff[3] = atan2(sin(x_diff[3]),cos(x_diff[3])); // normalize phi
    //angle normalization
    z_diff[1] = atan2(sin(z_diff[1]),cos(z_diff[1])); // normalize phi

    //Calculate measurement covariance matrix S
    S += (weights_(i) * z_diff * (z_diff.transpose()));
    //calculate cross correlation matrix
    Tc += (weights_(i) * x_diff * (z_diff.transpose()));
  }

  //measurement noise matrix R
  MatrixXd R = MatrixXd(n_z,n_z);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;

  // Add measurement noise R
  S += R;

  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;
  //angle normalization
  z_diff[1] = atan2(sin(z_diff[1]),cos(z_diff[1])); // normalize phi

  //update state mean and covariance matrix
  x_ += (K * z_diff);
  P_ -= (K * S * K.transpose());
}