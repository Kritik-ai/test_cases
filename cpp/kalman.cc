class KalmanFilter {
public:
    void Yiorderfilter(float angle_m, float gyro_m, float dt, float K1);
    double Kalman_Filter(double angle_m, double gyro_m, double dt,
        double Q_angle, double Q_gyro, double R_angle, double C_0);
    double angle;
    float angle6;

private:
    double angle_err;
    double q_bias; // gyroscope drift
    double Pdot[4] = { 0, 0, 0, 0 };
    double P[2][2] = { { 0, 0 }, { 0, 0 } };
    double PCt_0, PCt_1, E, K_0, K_1, t_0, t_1;
    double angle_dot;
};

void KalmanFilter::Yiorderfilter(
    float angle_m, float gyro_m, float dt, float K1)
{
    angle6 = K1 * angle_m + (1 - K1) * (angle6 + gyro_m * dt);
}

double KalmanFilter::Kalman_Filter(double angle_m, double gyro_m, double dt,
    double Q_angle, double Q_gyro, double R_angle, double C_0)
{
    angle += (gyro_m - q_bias) * dt; // Prior estimate
    angle_err = angle_m - angle;

    Pdot[0] = Q_angle - P[0][1] - P[1][0]
        + dt * P[1][1]; // Differential of azimuth error covariance.
    Pdot[1] = -P[1][1];
    Pdot[2] = -P[1][1];
    Pdot[3] = Q_gyro;

    P[0][0] += Pdot[0] * dt; // The integral of the covariance differential of
                             // the prior estimate error.
    P[0][1] += Pdot[1] * dt;
    P[1][0] += Pdot[2] * dt;
    P[1][1] += Pdot[3] * dt;

    // Intermediate variable of matrix multiplication.
    PCt_0 = C_0 * P[0][0];
    PCt_1 = C_0 * P[1][0];

    // Denominator
    E = R_angle + C_0 * PCt_0;

    // Gain value
    K_0 = PCt_0 / E;
    K_1 = PCt_1 / E;

    // Intermediate variable of matrix multiplication.
    t_0 = PCt_0;
    t_1 = C_0 * P[0][1];

    // Posterior estimation error covariance.
    P[2][2] -= K_0 * t_0;
    P[0][1] -= K_0 * t_1;
    P[1][0] -= K_1 * t_0;
    P[1][1] -= K_1 * t_1;

    angle+= K_0 * angle_err; // Posterior estimation; work out the optimal angle
    q_bias += K_1 * angle_err;   // Posterior estimation
    angle_dot = gyro_m - q_bias; // The differential value of the output value;
                                 // work out the optimal angular velocity

    return angle;
}
