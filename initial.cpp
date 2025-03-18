#include <iostream>
#include <string>
#include <random>

#define X_LOW -1.0
#define X_HIGH 1.0

#define V_LOW  0.0
#define V_HIGH 0.0

#define M_LOW  1.0
#define M_HIGH 1.0


int main(int argc, char **argv)
{
    int n = std::stoi(argv[1]);

    double x_min = (argc > 2) ? std::stof(argv[2]) : X_LOW;
    double x_max = (argc > 2) ? std::stof(argv[3]) : X_HIGH;

    double v_min = (argc > 2) ? std::stof(argv[4]) : V_LOW;
    double v_max = (argc > 2) ? std::stof(argv[5]) : V_HIGH;

    double m_min = (argc > 2) ? std::stof(argv[6]) : M_LOW;
    double m_max = (argc > 2) ? std::stof(argv[7]) : M_HIGH;

   	std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist_x(x_min, x_max);
    std::uniform_real_distribution<double> dist_v(v_min, v_max);
    std::uniform_real_distribution<double> dist_m(m_min, m_max);

    for (int i = 0; i < n; i++)
    {
        std::cout << dist_x(mt) << " " << dist_x(mt) << " " << dist_x(mt) << " " << dist_v(mt) << " " << dist_v(mt) << " " << dist_v(mt) << "  " << dist_m(mt) << " ";
    }
    return 0;
}

