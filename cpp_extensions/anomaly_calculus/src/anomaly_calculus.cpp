#include "anomaly_calculus.hpp"
#include <iostream>

AnomalyCalculus::AnomalyCalculus(float mse_threshold, float gaussian_sigma)
    : mse_threshold_(mse_threshold), gaussian_sigma_(gaussian_sigma) {}

float AnomalyCalculus::computePixelwiseMSE(const cv::Mat& raw_frame, const cv::Mat& recon_frame, cv::Mat& heatmap_out) {
    if (raw_frame.empty() || recon_frame.empty()) {
        return 0.0f;
    }

    cv::Mat raw_float, recon_float;
    
    // Normalize inputs to range [0.0, 1.0] CV_32F
    if (raw_frame.type() != CV_32F && raw_frame.type() != CV_32FC3) {
        raw_frame.convertTo(raw_float, CV_32F, 1.0 / 255.0);
    } else {
        raw_float = raw_frame.clone();
    }

    if (recon_frame.type() != CV_32F && recon_frame.type() != CV_32FC3) {
        recon_frame.convertTo(recon_float, CV_32F, 1.0 / 255.0);
    } else {
        recon_float = recon_frame.clone();
    }

    // Ensure matching spatial resolutions
    if (raw_float.size() != recon_float.size()) {
        cv::resize(recon_float, recon_float, raw_float.size(), 0, 0, cv::INTER_LINEAR);
    }

    // Convert multi-channel inputs to single channel grayscale if necessary
    if (raw_float.channels() > 1) {
        cv::cvtColor(raw_float, raw_float, cv::COLOR_BGR2GRAY);
    }
    if (recon_float.channels() > 1) {
        cv::cvtColor(recon_float, recon_float, cv::COLOR_BGR2GRAY);
    }

    // Compute pixel-wise squared difference matrix: (Raw - Recon)^2
    cv::Mat diff;
    cv::absdiff(raw_float, recon_float, diff);
    cv::multiply(diff, diff, heatmap_out);

    // Apply spatial smoothing filter to suppress isolated pixel noise
    if (gaussian_sigma_ > 0.0f) {
        cv::GaussianBlur(heatmap_out, heatmap_out, cv::Size(5, 5), gaussian_sigma_);
    }

    // Calculate mean MSE value across entire frame
    cv::Scalar mean_scalar = cv::mean(heatmap_out);
    return static_cast<float>(mean_scalar[0]);
}

float AnomalyCalculus::computeStructuralAnomaly(
    const cv::Mat& raw_frame,
    const cv::Mat& recon_frame,
    cv::Mat& heatmap_out,
    float w_dssim,
    float w_lncc,
    float w_mse,
    int window_size,
    float sigma,
    float c1,
    float c2
) {
    if (window_size <= 0 || window_size % 2 == 0) {
        throw std::invalid_argument("window_size must be a positive odd integer");
    }

    if (raw_frame.empty() || recon_frame.empty()) {
        return 0.0f;
    }

    cv::Mat raw_float, recon_float;
    if (raw_frame.type() != CV_32F && raw_frame.type() != CV_32FC3) {
        raw_frame.convertTo(raw_float, CV_32F, 1.0 / 255.0);
    } else {
        raw_float = raw_frame.clone();
    }

    if (recon_frame.type() != CV_32F && recon_frame.type() != CV_32FC3) {
        recon_frame.convertTo(recon_float, CV_32F, 1.0 / 255.0);
    } else {
        recon_float = recon_frame.clone();
    }

    if (raw_float.size() != recon_float.size()) {
        cv::resize(recon_float, recon_float, raw_float.size(), 0, 0, cv::INTER_LINEAR);
    }

    if (raw_float.channels() > 1) {
        cv::cvtColor(raw_float, raw_float, cv::COLOR_BGR2GRAY);
    }
    if (recon_float.channels() > 1) {
        cv::cvtColor(recon_float, recon_float, cv::COLOR_BGR2GRAY);
    }

    cv::Size ksize(window_size, window_size);

    cv::Mat mu_x, mu_y;
    cv::GaussianBlur(raw_float, mu_x, ksize, sigma, sigma);
    cv::GaussianBlur(recon_float, mu_y, ksize, sigma, sigma);

    cv::Mat mu_xx, mu_yy, mu_xy;
    cv::multiply(mu_x, mu_x, mu_xx);
    cv::multiply(mu_y, mu_y, mu_yy);
    cv::multiply(mu_x, mu_y, mu_xy);

    cv::Mat raw_sq, recon_sq, raw_recon;
    cv::multiply(raw_float, raw_float, raw_sq);
    cv::multiply(recon_float, recon_float, recon_sq);
    cv::multiply(raw_float, recon_float, raw_recon);

    cv::Mat sigma_xx, sigma_yy, sigma_xy;
    cv::GaussianBlur(raw_sq, sigma_xx, ksize, sigma, sigma);
    sigma_xx = sigma_xx - mu_xx;
    cv::max(sigma_xx, 0.0f, sigma_xx);

    cv::GaussianBlur(recon_sq, sigma_yy, ksize, sigma, sigma);
    sigma_yy = sigma_yy - mu_yy;
    cv::max(sigma_yy, 0.0f, sigma_yy);

    cv::GaussianBlur(raw_recon, sigma_xy, ksize, sigma, sigma);
    sigma_xy = sigma_xy - mu_xy;

    // 1. DSSIM
    cv::Mat num = (2.0f * mu_xy + c1).mul(2.0f * sigma_xy + c2);
    cv::Mat den = (mu_xx + mu_yy + c1).mul(sigma_xx + sigma_yy + c2);
    cv::Mat ssim;
    cv::divide(num, den, ssim);
    cv::min(cv::max(ssim, -1.0f), 1.0f, ssim);
    cv::Mat dssim = (1.0f - ssim) * 0.5f;

    // 2. Stabilized LNCC
    cv::Mat lncc_num = 2.0f * sigma_xy + c2;
    cv::Mat lncc_den = sigma_xx + sigma_yy + c2;
    cv::Mat lncc;
    cv::divide(lncc_num, lncc_den, lncc);
    cv::min(cv::max(lncc, -1.0f), 1.0f, lncc);
    cv::Mat dist_lncc = (1.0f - lncc) * 0.5f;

    // 3. Local MSE
    cv::Mat diff = raw_float - recon_float;
    cv::Mat diff_sq;
    cv::multiply(diff, diff, diff_sq);
    cv::Mat mse_local;
    cv::GaussianBlur(diff_sq, mse_local, ksize, sigma, sigma);

    // Composite Heatmap
    heatmap_out = w_dssim * dssim + w_lncc * dist_lncc + w_mse * mse_local;
    cv::min(cv::max(heatmap_out, 0.0f), 1.0f, heatmap_out);

    cv::Scalar mean_s = cv::mean(heatmap_out);
    return static_cast<float>(mean_s[0]);
}

void AnomalyCalculus::generateVisualHeatmap(const cv::Mat& mse_matrix, cv::Mat& visual_heatmap, bool apply_colormap) {
    if (mse_matrix.empty()) return;

    // Normalize MSE floating point matrix (0.0 - max_val) to uint8 (0 - 255)
    double min_val, max_val;
    cv::minMaxLoc(mse_matrix, &min_val, &max_val);

    double upper_bound = std::max(max_val, 0.25);
    cv::Mat normalized;
    mse_matrix.convertTo(normalized, CV_8U, 255.0 / upper_bound);

    if (apply_colormap) {
        cv::applyColorMap(normalized, visual_heatmap, cv::COLORMAP_JET);
    } else {
        visual_heatmap = normalized;
    }
}

void AnomalyCalculus::getAnomalyStats(const cv::Mat& mse_matrix, float& mean_mse, float& max_mse, cv::Point& max_loc) {
    if (mse_matrix.empty()) {
        mean_mse = 0.0f;
        max_mse = 0.0f;
        max_loc = cv::Point(0, 0);
        return;
    }

    double min_v, max_v;
    cv::minMaxLoc(mse_matrix, &min_v, &max_v, nullptr, &max_loc);
    cv::Scalar avg_s = cv::mean(mse_matrix);

    mean_mse = static_cast<float>(avg_s[0]);
    max_mse = static_cast<float>(max_v);
}
