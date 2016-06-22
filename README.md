## Attitude filter

Implementation of a quaternion based attitude filter as described in [1]. One difference is that this is an implementation of the Central Difference Kalman Filter instead of the more common Unscented Kalman Filter. For handling the quaternion updates, uses boxPlus and boxMinus manifold operations as introduced in [2] and also used in Rovio.

[1] John L. Crassidis and F. Landis Markley, "Unscented Filtering for Spacecraft Attitude Estimation"\
[2] C. Hertzberg et al., "Integrating generic sensor fusion algorithms with sound state representations through encapsulation of manifolds"
