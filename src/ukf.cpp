#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 10, 0,
          0, 0, 0, 0, 10;


    // storing h_laser_
    H_laser_ = MatrixXd(2, 5);
    H_laser_ << 1, 0, 0, 0, 0,
                0, 1, 0, 0, 0;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 2.5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = M_PI/4;

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
    
    // State dimension
    n_x_ = 5;
    
    // Augmented state dimension
    n_aug_ = 7;
    
    // Z measurement dimension
    n_z_ = 3;
    
    R_laser_ = MatrixXd(2, 2);
    R_laser_ << pow(std_laspx_,2), 0, 0, pow(std_laspy_,2);
    
    R_radar_ = MatrixXd(n_z_, n_z_);
    R_radar_.fill(0.0);
    R_radar_(0,0)=pow(std_radr_,2);
    R_radar_(1,1)=pow(std_radphi_,2);
    R_radar_(2,2)=pow(std_radrd_,2);
    
    
    // Sigma point spreading parameter
    lambda_ = 3 - n_aug_;
    
    weights_ = VectorXd(2*n_aug_+1);
    
    weights_(0) = lambda_/(lambda_+n_aug_);
    for (unsigned int i=1; i<2*n_aug_+1; i++){
        weights_(i) = 1/(2*(lambda_+n_aug_));
    }
    
    // For storing sigma points
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
    
}

UKF::~UKF() {}

double UKF::normalize_angle(double angle){
    while (angle>M_PI){angle-=2*M_PI;}
    while (angle<-M_PI){angle+=2*M_PI;}
    return angle;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    
    if (!is_initialized_) {
        time_us_= meas_package.timestamp_;
        float x0 = meas_package.raw_measurements_[0];
        float x1 = meas_package.raw_measurements_[1];
        
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            x_ << x0*cos(x1), x0*sin(x1), 0, 0, 0;
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            x_ << x0, x1, 0, 0, 0;
        }
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }
    
    
    double dt = (meas_package.timestamp_-time_us_)/1000000.0;
    time_us_ = meas_package.timestamp_;
    
    if (meas_package.sensor_type_==MeasurementPackage::RADAR){
        if (use_radar_){
            Prediction(dt);
            UpdateRadar(meas_package);
        }
    }else if (meas_package.sensor_type_==MeasurementPackage::LASER){
        if (use_laser_){
            Prediction(dt);
            UpdateLidar(meas_package);
        }
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double dt) {
    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);
    
    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    
    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    
    //set the first 5 values to be the exisiting mean
    x_aug.head(n_x_) = x_;
    // last two (representing noise) have zero mean
    x_aug.segment(n_x_, 2) << 0, 0;
    
    // top left (5x5) of augmented covariance matrix is just P
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    // bottom right (2x2)
    P_aug.bottomRightCorner(2,2) << pow(std_a_,2), 0, 0, pow(std_yawdd_,2);

    MatrixXd A = P_aug.llt().matrixL();
    Xsig_aug.col(0) = x_aug;

    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i+1)     = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
    }

    //create matrix with predicted sigma points as columns
    Xsig_pred_.fill(0.0);
    for (unsigned int i=0; i<Xsig_pred_.cols(); i++){
        
        float px = Xsig_aug(0,i);
        float py = Xsig_aug(1,i);
        float v = Xsig_aug(2,i);
        float psi = Xsig_aug(3,i);
        float psi_dot = Xsig_aug(4,i);
        float nu_a = Xsig_aug(5,i);
        float nu_psi = Xsig_aug(6,i);
        
        //psi = normalize_angle(psi);
       
        float px_out;
        float py_out;
        if (fabs(psi_dot)>0.0001){
            px_out = px + v/psi_dot*(sin(psi+psi_dot*dt)-sin(psi))+0.5*pow(dt,2)*cos(psi)*nu_a;
            py_out = py + v/psi_dot*(-cos(psi+psi_dot*dt)+cos(psi))+0.5*pow(dt,2)*sin(psi)*nu_a;
        }else{
            px_out = px + v*cos(psi)*dt+0.5*pow(dt,2)*cos(psi)*nu_a;
            py_out = py + v*sin(psi)*dt+0.5*pow(dt,2)*sin(psi)*nu_a;
        }
        
        float v_out = v + dt*nu_a;
        
        float psi_out = psi+psi_dot*dt + 0.5*pow(dt,2)*nu_psi;
        float psi_dot_out = psi_dot + dt*nu_psi;
        
        Xsig_pred_.col(i) << px_out,
                             py_out,
                             v_out,
                             psi_out,
                             psi_dot_out;
    }
    
    x_.fill(0.0);
    P_.fill(0.0);
    for (unsigned int i=0; i<Xsig_pred_.cols(); i++){
        x_+=weights_(i)*Xsig_pred_.col(i);
    }

    for (unsigned int i=0; i<Xsig_pred_.cols(); i++){
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        x_diff(3)=normalize_angle(x_diff(3));
        P_+=weights_(i)*x_diff*x_diff.transpose();    
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    
    VectorXd z = VectorXd(2);
    z << meas_package.raw_measurements_[0],
         meas_package.raw_measurements_[1];
    
    VectorXd z_pred = H_laser_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_laser_.transpose();
    MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
    MatrixXd K = P_ * Ht * S.inverse();
    
    //new estimate
    
    x_ = x_ + (K * y);
    MatrixXd I = MatrixXd::Identity(n_x_, n_x_);
    
    P_ = (I - K * H_laser_) * P_;
    
    x_(3)=normalize_angle(x_(3));
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    
    VectorXd z = VectorXd(n_z_);
    z << meas_package.raw_measurements_[0],
         meas_package.raw_measurements_[1],
         meas_package.raw_measurements_[2];
    
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);
    
    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z_);
    
    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z_,n_z_);
    
    //transform sigma points into measurement space
    for (unsigned int i=0; i<Zsig.cols(); i++){
        float px = Xsig_pred_(0,i);
        float py = Xsig_pred_(1,i);
        float v = Xsig_pred_(2,i);
        float psi = Xsig_pred_(3,i);
        float psi_dot = Xsig_pred_(4,i);
        
        float dist = sqrt(pow(px,2)+pow(py,2));
        float angle = atan2(py, px);
        angle = normalize_angle(angle);
        float dist_dot = (px*cos(psi)*v+py*sin(psi)*v)/dist;
        
        Zsig.col(i) << dist, angle, dist_dot;
    }
    
    //calculate mean predicted measurement
    z_pred.fill(0.0);
    for (unsigned int i=0; i<Zsig.cols(); i++){
        z_pred+=weights_(i)*Zsig.col(i);
    }
    
    //calculate measurement covariance matrix S
    S.fill(0.0);
    for (unsigned int i=0; i<Zsig.cols(); i++){
        S+=weights_(i)*(Zsig.col(i)-z_pred)*(Zsig.col(i)-z_pred).transpose();
    }
    S+=R_radar_;
    
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z_);
    
    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (unsigned int i=0; i<Zsig.cols(); i++){
        
        MatrixXd x_diff = Xsig_pred_.col(i)-x_;
        MatrixXd z_diff = Zsig.col(i)-z_pred;
        
        x_diff(3) = normalize_angle(x_diff(3));
        z_diff(1) = normalize_angle(z_diff(1));

        Tc+=weights_(i)*x_diff*z_diff.transpose();
    }
    //calculate Kalman gain K;
    MatrixXd K = Tc*S.inverse();
    
    //update state mean and covariance matrix
    x_ = x_ + K*(z-z_pred);
    P_ = P_ - K*S*K.transpose();
    
    x_(3)=normalize_angle(x_(3));
}
