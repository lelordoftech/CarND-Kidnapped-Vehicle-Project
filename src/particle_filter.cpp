/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 3;

  default_random_engine gen;

  // Creates a normal (Gaussian) distribution.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++)
  {
    struct Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;

    particles.push_back(particle);
    weights.push_back(1);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  for (int i = 0; i < num_particles; i++)
  {
    double theta0 = particles[i].theta;
    double theta_dot = yaw_rate*delta_t;
    double new_x;
    double new_y;
    double new_theta;

    if (yaw_rate == 0) // Just go straight
    {
      new_x = particles[i].x + velocity*delta_t*cos(theta0);
      new_y = particles[i].x + velocity*delta_t*sin(theta0);
      new_theta = theta0;
    }
    else
    {
      new_x = particles[i].x + velocity/yaw_rate * (sin(theta0+theta_dot) - sin(theta0));
      new_y = particles[i].y + velocity/yaw_rate * (cos(theta0) - cos(theta0+theta_dot));
      new_theta = theta0 + theta_dot;
    }

    // Creates a normal (Gaussian) distribution.
    normal_distribution<double> dist_x(new_x, std_pos[0]);
    normal_distribution<double> dist_y(new_y, std_pos[1]);
    normal_distribution<double> dist_theta(new_theta, std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

  for (int i = 0; i < observations.size(); i++)
  {
    int closest_id = 0;
    double min_dist = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y);

    for (int j = 0; j < predicted.size(); j++)
    {
      double curr_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if (curr_dist <= min_dist)
      {
        min_dist = curr_dist;
        closest_id = j;
      }
    }

    observations[i] = predicted[closest_id];
  }
}

void ParticleFilter::transformedObservation(Particle particle, LandmarkObs& observation)
{
  // rotation
  double theta = particle.theta;
  double X = observation.x*cos(theta) - observation.y*sin(theta);
  double Y = observation.x*sin(theta) + observation.y*cos(theta);
  // translation
  observation.x = particle.x + X;
  observation.y = particle.y + Y;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks)
{
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < num_particles; i++)
  {
    std::vector<LandmarkObs> observations_trans;

    // Step 1: transform each observation marker from the vehicle's coordinates to the map's coordinates, in respect to our particle
    for (int j = 0; j < observations.size(); j++)
    {
      struct LandmarkObs observation;
      observation = observations[j];
      transformedObservation(particles[i], observation);
      observations_trans.push_back(observation);
    }

    // Step 2: associate each transformed observation with a land mark identifier
    std::vector<LandmarkObs> predicted_obs = observations_trans;
    std::vector<LandmarkObs> landmark_obs;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      struct LandmarkObs landmark;
      landmark.id = map_landmarks.landmark_list[j].id_i;
      landmark.x = map_landmarks.landmark_list[j].x_f;
      landmark.y = map_landmarks.landmark_list[j].y_f;

      double range = dist(particles[i].x, particles[i].y, landmark.x, landmark.y);
      if (range <= sensor_range) // In sensor range
      {
        landmark_obs.push_back(landmark);
      }
    }

    dataAssociation(landmark_obs, predicted_obs);

    // Step 3: calculate the particle's final weight
    long double weight = 1.0;
    double weight_part = 1.0;
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    for (int j = 0; j < predicted_obs.size(); j++)
    {
      if (predicted_obs[j].id > 0)
      {
        weight_part = 1/(2*M_PI*std_landmark[0]*std_landmark[1]) * 
                        exp(-(pow(observations_trans[j].x-predicted_obs[j].x, 2)/(2*pow(std_landmark[0],2)) + 
                          pow(observations_trans[j].y-predicted_obs[j].y, 2)/(2*pow(std_landmark[1],2))));

        if (weight_part > 0)
        {
          weight *= weight_part;
        }
      }
      associations.push_back(predicted_obs[j].id);
      sense_x.push_back(observations_trans[j].x);
      sense_y.push_back(observations_trans[j].y);
    }
    particles[i].weight = (double)weight;

    particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
    weights[i] = particles[i].weight; // Update weights
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());

  vector<Particle> resample_particles;

  for (int i = 0; i < num_particles; i++)
  {
    resample_particles.push_back(particles[distribution(gen)]);
  }

  particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
