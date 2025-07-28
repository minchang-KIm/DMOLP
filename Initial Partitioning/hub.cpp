#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

#include "hub.hpp"

using namespace std;

vector<int> find_hub_nodes(const unordered_map<int, int> &global_degree) {
    vector<pair<int, int>> sorted_degree(global_degree.begin(), global_degree.end());

    sort(sorted_degree.begin(), sorted_degree.end(), [](const auto &a, const auto &b) {
        return a.second < b.second;
    });

    int N = sorted_degree.size();
    vector<double> cum_x(N), cum_y(N);

    double total_deg = 0.0;
    for (const auto &[_, deg] : sorted_degree)
        total_deg += deg;

    double sum_deg = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_deg += sorted_degree[i].second;
        cum_x[i] = (double)(i + 1) / N;
        cum_y[i] = sum_deg / total_deg;
    }

    double x1 = cum_x[N - 1];
    double y1 = cum_y[N - 1];
    double x0 = cum_x[N - 2];
    double y0 = cum_y[N - 2];
    double slope = (y1 - y0) / (x1 - x0);

    double x_intercept = x1 - y1 / slope;

    vector<int> hub_nodes;
    for (int i = N - 1; i >= 0; --i) {
        if (cum_x[i] > x_intercept) hub_nodes.push_back(sorted_degree[i].first);
        else break;
    }

    return hub_nodes;
}

vector<int> find_landmarks(const unordered_map<int, int> &global_degree) {
    int N = global_degree.size();
    int K = max(1, (int)log10(N));

    vector<pair<int, int>> sorted_degree(global_degree.begin(), global_degree.end());
    sort(sorted_degree.begin(), sorted_degree.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;
    });

    vector<int> landmarks;
    for (int i = 0; i < K && i < sorted_degree.size(); ++i)
        landmarks.push_back(sorted_degree[i].first);

    return landmarks;
}