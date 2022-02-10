#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#include "cudaSift.h"

int ImproveHomography(SiftData &data, float *homography, int numLoops, float minScore,
                      float maxAmbiguity, float thresh) {
#ifdef MANAGEDMEM
  SiftPoint *mpts = data.m_data;
#else
  if (data.h_data == NULL) return 0;
  SiftPoint *mpts = data.h_data;
#endif
  float limit = thresh * thresh;
  int numPts = data.numPts;
  Eigen::MatrixXd M(8, 8);
  Eigen::VectorXd A(8);
  Eigen::VectorXd X(8);
  Eigen::VectorXd Y(8);
  for (int i = 0; i < 8; i++) A(i) = homography[i] / homography[8];
  for (int loop = 0; loop < numLoops; loop++) {
    M = Eigen::MatrixXd::Zero(8, 8);
    X = Eigen::VectorXd::Zero(8);
    for (int i = 0; i < numPts; i++) {
      SiftPoint &pt = mpts[i];
      if (pt.score < minScore || pt.ambiguity > maxAmbiguity) continue;
      float den = A(6) * pt.xpos + A(7) * pt.ypos + 1.0f;
      float dx = (A(0) * pt.xpos + A(1) * pt.ypos + A(2)) / den - pt.match_xpos;
      float dy = (A(3) * pt.xpos + A(4) * pt.ypos + A(5)) / den - pt.match_ypos;
      float err = dx * dx + dy * dy;
      float wei = (err < limit ? 1.0f : 0.0f);  // limit / (err + limit);
      Y(0) = pt.xpos;
      Y(1) = pt.ypos;
      Y(2) = 1.0;
      Y(3) = Y(4) = Y(5) = 0.0;
      Y(6) = -pt.xpos * pt.match_xpos;
      Y(7) = -pt.ypos * pt.match_xpos;
      for (int c = 0; c < 8; c++)
        for (int r = 0; r < 8; r++) M(r, c) += (Y(c) * Y(r) * wei);
      X += (Y * pt.match_xpos * wei);
      Y(0) = Y(1) = Y(2) = 0.0;
      Y(3) = pt.xpos;
      Y(4) = pt.ypos;
      Y(5) = 1.0;
      Y(6) = -pt.xpos * pt.match_ypos;
      Y(7) = -pt.ypos * pt.match_ypos;
      for (int c = 0; c < 8; c++)
        for (int r = 0; r < 8; r++) M(r, c) += (Y(c) * Y(r) * wei);
      X += (Y * pt.match_ypos * wei);
    }
    // cv::solve(M, X, A, cv::DECOMP_CHOLESKY);
    A = M.colPivHouseholderQr().solve(X);
  }
  int numfit = 0;
  for (int i = 0; i < numPts; i++) {
    SiftPoint &pt = mpts[i];
    float den = A(6) * pt.xpos + A(7) * pt.ypos + 1.0;
    float dx = (A(0) * pt.xpos + A(1) * pt.ypos + A(2)) / den - pt.match_xpos;
    float dy = (A(3) * pt.xpos + A(4) * pt.ypos + A(5)) / den - pt.match_ypos;
    float err = dx * dx + dy * dy;
    if (err < limit) numfit++;
    pt.match_error = sqrt(err);
  }
  for (int i = 0; i < 8; i++) homography[i] = A(i);
  homography[8] = 1.0f;
  return numfit;
}
