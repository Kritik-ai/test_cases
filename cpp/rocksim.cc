#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <thread>
#include <vector>

#define RANGE(VAR, END)                                                        \
  int VAR = 0;                                                                 \
  VAR < END;                                                                   \
  ++VAR
using namespace std;
typedef std::pair<int, int> coord;

void draw(std::vector<coord> &rock, set<coord> &world, int hmax) {
  std::vector<std::vector<char>> canv(hmax, vector<char>(7, '.'));

  for (auto r : rock) {
    canv[r.second][r.first] = '@';
  }

  for (auto r : world) {
    canv[r.second][r.first] = '#';
  }

  for (auto it = canv.rbegin(); it != canv.rend(); ++it) {
    printf("%s\n", (*it).data());
  }
  printf("-------\n\n");
}

std::vector<char> parse(std::string filename) {
  std::fstream fs(filename);

  std::string line;
  getline(fs, line);

  return std::vector<char>(line.begin(), line.end());
}

inline bool xmov(vector<coord> &rock, int dx, int width, set<coord> &world) {
  for (auto &r : rock) {
    if ((r.first + dx) >= width || (r.first + dx) < 0 ||
        world.contains(coord(r.first + dx, r.second))) {
      return false;
    }
  }

  std::for_each(rock.begin(), rock.end(),
                [dx](coord &pos) { pos.first += dx; });
  return true;
}

inline void ymov(vector<coord> &rock, int dy) {
  std::for_each(rock.begin(), rock.end(),
                [dy](coord &pos) { pos.second += dy; });
}

inline bool check_y_collision(vector<coord> &rock, set<coord> &world) {
  for (auto &r : rock) {
    if (world.contains(r)) {
      return true;
    }
  }

  return false;
}

inline int find_ymax(int ymax, vector<coord> rock) {
  rock.push_back(coord(0, ymax));
  return std::max_element(rock.begin(), rock.end(),
                          [](const coord c1, const coord c2) {
                            return c1.second < c2.second;
                          })
      ->second;
}

int rock_sim(vector<coord> rock, set<coord> &world, vector<char> &winds,
             int width, int ymax, int &wind_idx) {
  const int xstart = 2;
  const int ystart = 4;

  ymov(rock, ymax + ystart);
  xmov(rock, xstart, width, world);

  vector<coord> rock_prev;
  while (1) {
    // draw(rock, world, ymax + 10);
    switch (winds[wind_idx % winds.size()]) {
    case '<':
      xmov(rock, -1, width, world);
      break;
    case '>':
      xmov(rock, 1, width, world);
      break;
    }
    wind_idx++;

    rock_prev = rock;
    ymov(rock, -1);

    if (check_y_collision(rock, world)) {
      for (auto r : rock_prev) {
        world.insert(r);
      }

      return find_ymax(ymax, rock_prev);
    }
  }
}

void sim(vector<vector<coord>> &rocks, set<coord> &world, vector<char> &winds,
         int width = 7, int num_sims = 2022) {
  vector<string> names = {"flat", "cross", "L", "line", "cube"};
  int ymax = 0;
  int wind_idx = 0;
  for (RANGE(i, num_sims)) {
    // printf("%s\n", names[i % rocks.size()].data());
    ymax =
        rock_sim(rocks[i % rocks.size()], world, winds, width, ymax, wind_idx);
    // printf("%i\n", ymax);
  }

  std::cout << "Part 1: " << ymax << "\n\r";
}

void part1(vector<char> &winds) {
  vector<coord> flat = {{0, 0}, {1, 0}, {2, 0}, {3, 0}};
  vector<coord> cross = {{1, 0}, {0, 1}, {1, 1}, {2, 1}, {1, 2}};
  vector<coord> L = {{0, 0}, {1, 0}, {2, 0}, {2, 1}, {2, 2}};
  vector<coord> line = {{0, 0}, {0, 1}, {0, 2}, {0, 3}};
  vector<coord> cube = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

  vector<vector<coord>> rocks = {flat, cross, L, line, cube};

  set<coord> world = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}};

  sim(rocks, world, winds);
}

int main() {
  auto winds = parse("../inp_1.txt");
  part1(winds);
}
