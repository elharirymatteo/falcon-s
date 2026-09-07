import warp as wp
import random

@wp.func
def add_gaussian_noise(value: wp.float32, noise_std: wp.float32, seed: wp.int32, offset: wp.int32) -> wp.float32:
    """Add Gaussian noise to a scalar value"""
    state = wp.rand_init(seed, offset)
    u1 = wp.randf(state)
    u2 = wp.randf(state)
    z = wp.sqrt(-2.0 * wp.log(u1)) * wp.cos(2.0 * wp.PI * u2)
    return value + noise_std * z

@wp.func
def add_gaussian_noise_vec3(value: wp.vec3f, noise_std: wp.float32, seed: wp.int32, offset: wp.int32) -> wp.vec3f:
    """Add Gaussian noise to a vec3 value"""
    noisy_x = add_gaussian_noise(value[0], noise_std, seed, offset)
    noisy_y = add_gaussian_noise(value[1], noise_std, seed, offset + 1234)
    noisy_z = add_gaussian_noise(value[2], noise_std, seed, offset + 2345)
    return wp.vec3f(noisy_x, noisy_y, noisy_z)

@wp.func
def add_gaussian_noise_quat(value: wp.quatf, noise_std: wp.float32, seed: wp.int32, offset: wp.int32) -> wp.quatf:
    """Add Gaussian noise to quaternion components and renormalize"""
    noisy_q1 = add_gaussian_noise(value[0], noise_std, seed, offset + 1010)
    noisy_q2 = add_gaussian_noise(value[1], noise_std, seed, offset + 3456)
    noisy_q3 = add_gaussian_noise(value[2], noise_std, seed, offset + 4321)
    noisy_q0 = add_gaussian_noise(value[3], noise_std, seed, offset + 1111)
    noisy_quat = wp.quatf(noisy_q1, noisy_q2, noisy_q3, noisy_q0)
    return wp.normalize(noisy_quat)

@wp.func
def apply_sensor_limits(value: wp.float32, min_val: wp.float32, max_val: wp.float32, resolution: wp.float32) -> wp.float32:
    """Apply min/max limits and resolution quantization"""
    clamped_value = wp.clamp(value, min_val, max_val)
    if resolution > 0.0:
        quantized_value = wp.round(clamped_value / resolution) * resolution
        return wp.clamp(quantized_value, min_val, max_val)
    return clamped_value

@wp.func
def apply_sensor_limits_vec3(value: wp.vec3f, min_val: wp.vec3f, max_val: wp.vec3f, resolution: wp.float32) -> wp.vec3f:
    """Apply sensor limits to vec3"""
    return wp.vec3f(
        apply_sensor_limits(value[0], min_val[0], max_val[0], resolution),
        apply_sensor_limits(value[1], min_val[1], max_val[1], resolution),
        apply_sensor_limits(value[2], min_val[2], max_val[2], resolution)
    )

@wp.func
def check_sampling_time(current_time: wp.float32, last_sample_time: wp.float32, sampling_period: wp.float32) -> bool:
    """Check if enough time has passed for next sample"""
    # Always sample at t=0
    if current_time == 0.0:
        return True
    
    # Check if enough time has passed since the last sample
    time_since_last = current_time - last_sample_time
    return time_since_last >= (sampling_period - 1e-6)  # Small tolerance for floating point

@wp.kernel
def write_to_delay_buffer_vec3(
    measurement: wp.array(dtype=wp.vec3f),
    timestamp: wp.float32,
    buffer: wp.array2d(dtype=wp.vec3f),
    shared_time_buffer: wp.array(dtype=wp.float32),
    write_index: wp.array(dtype=wp.int32),
    buffer_size: wp.int32
) -> None:
    """Write measurement to circular delay buffer"""
    tid = wp.tid()
    
    idx = write_index[tid]
    buffer[tid, idx] = measurement[tid]
    shared_time_buffer[tid * buffer_size + idx] = timestamp
    
    # Update write index (circular)
    write_index[tid] = (idx + 1) % buffer_size

@wp.kernel
def read_from_delay_buffer_vec3(
    target_time: wp.float32,
    shared_time_buffer: wp.array(dtype=wp.float32),
    buffer: wp.array2d(dtype=wp.vec3f),
    write_index: wp.array(dtype=wp.int32),
    buffer_size: wp.int32,
    delayed_measurement: wp.array(dtype=wp.vec3f),
    valid_output: wp.array(dtype=bool)
) -> None:
    """Read delayed measurement from buffer"""
    tid = wp.tid()
    
    if target_time <= 0.0:
        # Return first measurement (t=0)
        delayed_measurement[tid] = buffer[tid, 0]
        valid_output[tid] = True
        return
    
    # Declare as dynamic variables
    best_index = int(0)
    best_diff = float(wp.abs(shared_time_buffer[tid * buffer_size] - target_time))
    
    for i in range(buffer_size):
        diff = float(wp.abs(shared_time_buffer[tid * buffer_size + i] - target_time))
        if diff < best_diff:
            best_diff = diff
            best_index = i
    
    delayed_measurement[tid] = buffer[tid, best_index]
    valid_output[tid] = True

@wp.kernel
def read_from_delay_buffer_quat(
    target_time: wp.float32,
    shared_time_buffer: wp.array(dtype=wp.float32),
    buffer: wp.array2d(dtype=wp.quatf),
    write_index: wp.array(dtype=wp.int32),
    buffer_size: wp.int32,
    delayed_measurement: wp.array(dtype=wp.quatf),
    valid_output: wp.array(dtype=bool)
) -> None:
    """Read delayed quaternion measurement from buffer"""
    tid = wp.tid()
    
    if target_time <= 0.0:
        # Return first measurement (t=0)
        delayed_measurement[tid] = buffer[tid, 0]
        valid_output[tid] = True
        return
    
    # Declare as dynamic variables  
    best_index = int(0)
    best_diff = float(wp.abs(shared_time_buffer[tid * buffer_size] - target_time))
    
    for i in range(buffer_size):
        diff = float(wp.abs(shared_time_buffer[tid * buffer_size + i] - target_time))
        if diff < best_diff:
            best_diff = diff
            best_index = i
    
    delayed_measurement[tid] = buffer[tid, best_index]
    valid_output[tid] = True

@wp.kernel
def write_to_delay_buffer_quat(
    measurement: wp.array(dtype=wp.quatf),
    timestamp: wp.float32,
    buffer: wp.array2d(dtype=wp.quatf),
    shared_time_buffer: wp.array(dtype=wp.float32),
    write_index: wp.array(dtype=wp.int32),
    buffer_size: wp.int32
) -> None:
    """Write quaternion measurement to circular delay buffer"""
    tid = wp.tid()
    
    idx = write_index[tid]
    buffer[tid, idx] = measurement[tid]
    shared_time_buffer[tid * buffer_size + idx] = timestamp
    
    # Update write index (circular)
    write_index[tid] = (idx + 1) % buffer_size

@wp.kernel
def sense_gps_position(
    true_position: wp.array(dtype=wp.vec3f),
    seed: wp.int32,
    noise_std: wp.float32,
    bias: wp.vec3f,
    scale_factor: wp.float32,
    min_value: wp.vec3f,
    max_value: wp.vec3f,
    resolution: wp.float32,
    enable_noise: bool,
    current_time: wp.float32,
    sampling_period: wp.float32,
    time_step: wp.int32,
    last_sample_time: wp.array(dtype=wp.float32),
    measured_position: wp.array(dtype=wp.vec3f),
    valid_measurement: wp.array(dtype=bool)
) -> None:
    """GPS position sensor with sampling rate control"""
    tid = wp.tid()
    
    should_sample = check_sampling_time(current_time, last_sample_time[tid], sampling_period)
    
    if should_sample:
        position = true_position[tid]
        position = scale_factor * position + bias

        if enable_noise:
            seed_with_time = seed + time_step
            position = add_gaussian_noise_vec3(position, noise_std, seed_with_time, tid)
        
        final_position = apply_sensor_limits_vec3(position, min_value, max_value, resolution)
        measured_position[tid] = final_position
        valid_measurement[tid] = True
        
        last_sample_time[tid] = current_time
    else:
        # Return last measurement when not sampling
        valid_measurement[tid] = False

@wp.kernel
def sense_gps_velocity(
    true_velocity: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    seed: wp.int32,
    noise_std: wp.float32,
    bias: wp.vec3f,
    scale_factor: wp.float32,
    min_value: wp.vec3f,
    max_value: wp.vec3f,
    resolution: wp.float32,
    enable_noise: bool,
    current_time: wp.float32,
    sampling_period: wp.float32,
    time_step: wp.int32,
    last_sample_time: wp.array(dtype=wp.float32),
    measured_velocity: wp.array(dtype=wp.vec3f),
    valid_measurement: wp.array(dtype=bool)
) -> None:
    """GPS velocity sensor with sampling rate control"""
    tid = wp.tid()
    
    should_sample = check_sampling_time(current_time, last_sample_time[tid], sampling_period)
    
    if should_sample:
        velocity = scale_factor * true_velocity[tid] + bias

        if enable_noise:
            seed_with_time = seed + time_step
            velocity = add_gaussian_noise_vec3(velocity, noise_std, seed_with_time, tid)
        
        final_velocity = apply_sensor_limits_vec3(velocity, min_value, max_value, resolution)
        measured_velocity[tid] = final_velocity
        valid_measurement[tid] = True
        
        last_sample_time[tid] = current_time
    else:
        valid_measurement[tid] = False

@wp.kernel
def sense_attitude(
    true_orientation: wp.array(dtype=wp.quatf),
    seed: wp.int32,
    noise_std: wp.float32,
    enable_noise: bool,
    current_time: wp.float32,
    sampling_period: wp.float32,
    time_step: wp.int32,
    last_sample_time: wp.array(dtype=wp.float32),
    measured_orientation: wp.array(dtype=wp.quatf),
    valid_measurement: wp.array(dtype=bool)
) -> None:
    """Attitude sensor with sampling rate control"""
    tid = wp.tid()
    
    should_sample = check_sampling_time(current_time, last_sample_time[tid], sampling_period)
    
    if should_sample:
        orientation = true_orientation[tid]
        
        if enable_noise:
            seed_with_time = seed + time_step
            orientation = add_gaussian_noise_quat(orientation, noise_std, seed_with_time, tid)
        
        measured_orientation[tid] = orientation
        valid_measurement[tid] = True    
        
        last_sample_time[tid] = current_time
    else:
        valid_measurement[tid] = False

@wp.kernel
def sense_gyroscope(
    true_angular_velocity: wp.array(dtype=wp.vec3f),
    seed: wp.int32,
    noise_std: wp.float32,
    bias: wp.vec3f,
    scaling_factor: wp.float32,
    min_value: wp.vec3f,
    max_value: wp.vec3f,
    resolution: wp.float32,
    enable_noise: bool,
    current_time: wp.float32,
    sampling_period: wp.float32,
    time_step: wp.int32,
    last_sample_time: wp.array(dtype=wp.float32),
    measured_angular_velocity: wp.array(dtype=wp.vec3f),
    valid_measurement: wp.array(dtype=bool)
) -> None:
    """Gyroscope sensor with sampling rate control"""
    tid = wp.tid()

    should_sample = check_sampling_time(current_time, last_sample_time[tid], sampling_period)

    if should_sample:
        angular_vel = scaling_factor * true_angular_velocity[tid] + bias

        if enable_noise:
            seed_with_time = seed + time_step
            angular_vel = add_gaussian_noise_vec3(angular_vel, noise_std, seed_with_time, tid)
        
        final_angular_vel = apply_sensor_limits_vec3(angular_vel, min_value, max_value, resolution)
        measured_angular_velocity[tid] = final_angular_vel
        valid_measurement[tid] = True
        
        last_sample_time[tid] = current_time
    else:
        valid_measurement[tid] = False

class SensorSystemWARP:
    """WARP-based sensor system for aircraft simulation"""
    
    def __init__(self, num_envs: int, device: str, sensor_config: dict, dt: float):
        self._num_envs = num_envs
        self._device = device
        self.config = sensor_config
        self.dt = dt
        
        self.time_step = 0
        self.warp_seed = None
        
        self.build_sensor_buffers()
        self.setup_sensor_parameters()
        self.setup_delay_buffers()
    
    def build_sensor_buffers(self):
        """Create WARP arrays for sensor outputs"""
        self.measured_position = wp.zeros((self._num_envs,), device=self._device, dtype=wp.vec3f)
        self.measured_velocity = wp.zeros((self._num_envs,), device=self._device, dtype=wp.vec3f)
        self.measured_orientation = wp.zeros((self._num_envs,), device=self._device, dtype=wp.quatf)
        self.measured_angular_velocity = wp.zeros((self._num_envs,), device=self._device, dtype=wp.vec3f)
        
        # Last sample time buffers
        self.gps_pos_last_sample_time = wp.full((self._num_envs,), -1.0, device=self._device, dtype=wp.float32)
        self.gps_vel_last_sample_time = wp.full((self._num_envs,), -1.0, device=self._device, dtype=wp.float32)
        self.att_last_sample_time = wp.full((self._num_envs,), -1.0, device=self._device, dtype=wp.float32)
        self.gyro_last_sample_time = wp.full((self._num_envs,), -1.0, device=self._device, dtype=wp.float32)
        
        # Validity buffers
        self.gps_pos_valid = wp.zeros((self._num_envs,), device=self._device, dtype=bool)
        self.gps_vel_valid = wp.zeros((self._num_envs,), device=self._device, dtype=bool)
        self.att_valid = wp.zeros((self._num_envs,), device=self._device, dtype=bool)
        self.gyro_valid = wp.zeros((self._num_envs,), device=self._device, dtype=bool)
    
    def setup_sensor_parameters(self):
        """Extract sensor parameters from configuration and calculate buffer sizes"""
        # GPS position sensor
        gps_pos_config = self.config.get('navigation', {}).get('gps_position', {})
        self.gps_pos_noise_std = gps_pos_config.get('noise_std', 0.05)
        self.gps_pos_bias = wp.vec3f(*gps_pos_config.get('bias', [0.0, 0.0, 0.0]))
        self.gps_pos_scaling_factor = gps_pos_config.get('scaling_factor', 1.0)
        self.gps_pos_min_value = wp.vec3f(*gps_pos_config.get('min_value', [float('-inf'), float('-inf'), float('-inf')]))
        self.gps_pos_max_value = wp.vec3f(*gps_pos_config.get('max_value', [float('inf'), float('inf'), float('inf')]))
        self.gps_pos_resolution = gps_pos_config.get('resolution', 0.0)
        self.gps_pos_sampling_rate = gps_pos_config.get('sampling_rate', 100.0)
        self.gps_pos_sampling_period = 1.0 / self.gps_pos_sampling_rate
        self.gps_pos_enable_noise = gps_pos_config.get('enable_noise', False)
        self.gps_pos_delay = gps_pos_config.get('delay', 0.0)
        
        # Calculate buffer size based on delay and dt
        self.gps_pos_buffer_size = max(1, int(self.gps_pos_delay / self.dt) + 10) if self.gps_pos_delay > 0 else 1
        
        # GPS velocity sensor
        gps_vel_config = self.config.get('navigation', {}).get('gps_velocity', {})
        self.gps_vel_noise_std = gps_vel_config.get('noise_std', 0.05)
        self.gps_vel_bias = wp.vec3f(*gps_vel_config.get('bias', [0.0, 0.0, 0.0]))
        self.gps_vel_scaling_factor = gps_vel_config.get('scaling_factor', 1.0)
        self.gps_vel_min_value = wp.vec3f(*gps_vel_config.get('min_value', [float('-inf'), float('-inf'), float('-inf')]))
        self.gps_vel_max_value = wp.vec3f(*gps_vel_config.get('max_value', [float('inf'), float('inf'), float('inf')]))
        self.gps_vel_resolution = gps_vel_config.get('resolution', 0.0)
        self.gps_vel_sampling_rate = gps_vel_config.get('sampling_rate', 100.0)
        self.gps_vel_sampling_period = 1.0 / self.gps_vel_sampling_rate
        self.gps_vel_enable_noise = gps_vel_config.get('enable_noise', False)
        self.gps_vel_delay = gps_vel_config.get('delay', 0.0)
        
        self.gps_vel_buffer_size = max(1, int(self.gps_vel_delay / self.dt) + 10) if self.gps_vel_delay > 0 else 1

        # Attitude sensor
        att_config = self.config.get('imu', {}).get('attitude_sensor', {})
        self.att_noise_std = att_config.get('noise_std', 0.01)
        self.att_sampling_rate = att_config.get('sampling_rate', 100.0)
        self.att_sampling_period = 1.0 / self.att_sampling_rate
        self.att_enable_noise = att_config.get('enable_noise', False)
        self.att_delay = att_config.get('delay', 0.0)
        
        self.att_buffer_size = max(1, int(self.att_delay / self.dt) + 10) if self.att_delay > 0 else 1

        # Gyroscope
        gyro_config = self.config.get('imu', {}).get('gyroscope', {})
        self.gyro_noise_std = gyro_config.get('noise_std', 0.01)
        self.gyro_bias = wp.vec3f(*gyro_config.get('bias', [0.0, 0.0, 0.0]))
        self.gyro_scaling_factor = gyro_config.get('scaling_factor', 1.0)
        self.gyro_min_value = wp.vec3f(*gyro_config.get('min_value', [float('-inf'), float('-inf'), float('-inf')]))
        self.gyro_max_value = wp.vec3f(*gyro_config.get('max_value', [float('inf'), float('inf'), float('inf')]))
        self.gyro_resolution = gyro_config.get('resolution', 0.0)
        self.gyro_sampling_rate = gyro_config.get('sampling_rate', 100.0)
        self.gyro_sampling_period = 1.0 / self.gyro_sampling_rate
        self.gyro_enable_noise = gyro_config.get('enable_noise', False)
        self.gyro_delay = gyro_config.get('delay', 0.0)
        
        self.gyro_buffer_size = max(1, int(self.gyro_delay / self.dt) + 10) if self.gyro_delay > 0 else 1

    def setup_delay_buffers(self):
        """Create delay buffers with calculated sizes"""
        # GPS position delay buffers
        if self.gps_pos_delay > 0:
            self.gps_pos_delay_buffer = wp.zeros((self._num_envs, self.gps_pos_buffer_size), device=self._device, dtype=wp.vec3f)
        else:
            self.gps_pos_delay_buffer = None  # Direct passthrough
        
        # GPS velocity delay buffers
        if self.gps_vel_delay > 0:
            self.gps_vel_delay_buffer = wp.zeros((self._num_envs, self.gps_vel_buffer_size), device=self._device, dtype=wp.vec3f)
        else:
            self.gps_vel_delay_buffer = None  # Direct passthrough
        
        # Attitude delay buffers
        if self.att_delay > 0:
            self.att_delay_buffer = wp.zeros((self._num_envs, self.att_buffer_size), device=self._device, dtype=wp.quatf)
        else:
            self.att_delay_buffer = None  # Direct passthrough
        
        # Gyroscope delay buffers
        if self.gyro_delay > 0:
            self.gyro_delay_buffer = wp.zeros((self._num_envs, self.gyro_buffer_size), device=self._device, dtype=wp.vec3f)
        else:
            self.gyro_delay_buffer = None  # Direct passthrough

        # Shared time and write index buffers
        max_buffer_size = max(self.gps_pos_buffer_size, self.gps_vel_buffer_size, self.att_buffer_size, self.gyro_buffer_size)
        self.shared_time_buffer = wp.zeros((self._num_envs * max_buffer_size,), device=self._device, dtype=wp.float32)
        self.shared_write_index = wp.zeros((self._num_envs,), device=self._device, dtype=wp.int32)

    def seed(self, seed=None):
        """Seed the sensor system"""
        if seed is not None:
            self.warp_seed = wp.int32(seed)
        else:
            random_seed = random.randint(0, 2**29)
            self.warp_seed = wp.int32(random_seed)

    def sense_all_sensors(self, aircraft_state: dict, dt: float):
        current_time = self.time_step * dt
        
        # GPS Position
        wp.launch(
            kernel=sense_gps_position,
            dim=self._num_envs,
            inputs=[
                aircraft_state['position'],
                self.warp_seed,
                self.gps_pos_noise_std,
                self.gps_pos_bias,
                self.gps_pos_scaling_factor,
                self.gps_pos_min_value,
                self.gps_pos_max_value,
                self.gps_pos_resolution,
                self.gps_pos_enable_noise,
                current_time,
                self.gps_pos_sampling_period,
                wp.int32(self.time_step),
                self.gps_pos_last_sample_time,
                self.measured_position,
                self.gps_pos_valid
            ],
            device=self._device
        )
        
        # GPS position delay processing (always run)
        if self.gps_pos_delay_buffer is not None:
            wp.launch(
                kernel=write_to_delay_buffer_vec3,
                dim=self._num_envs,
                inputs=[
                    self.measured_position,
                    current_time,
                    self.gps_pos_delay_buffer,
                    self.shared_time_buffer,
                    self.shared_write_index,
                    wp.int32(self.gps_pos_buffer_size)
                ],
                device=self._device
            )
            
            delayed_time = current_time - self.gps_pos_delay
            wp.launch(
                kernel=read_from_delay_buffer_vec3,
                dim=self._num_envs,
                inputs=[
                    delayed_time,
                    self.shared_time_buffer,
                    self.gps_pos_delay_buffer,
                    self.shared_write_index,
                    wp.int32(self.gps_pos_buffer_size),
                    self.measured_position,
                    self.gps_pos_valid
                ],
                device=self._device
            )
        
        # GPS Velocity
        wp.launch(
            kernel=sense_gps_velocity,
            dim=self._num_envs,
            inputs=[
                aircraft_state['linear_vel'],
                aircraft_state['orientation'],
                self.warp_seed,
                self.gps_vel_noise_std,
                self.gps_vel_bias,
                self.gps_vel_scaling_factor,
                self.gps_vel_min_value,
                self.gps_vel_max_value,
                self.gps_vel_resolution,
                self.gps_vel_enable_noise,
                current_time,
                self.gps_vel_sampling_period,
                wp.int32(self.time_step),
                self.gps_vel_last_sample_time,
                self.measured_velocity,
                self.gps_vel_valid
            ],
            device=self._device
        )
        
        # GPS velocity delay processing
        if self.gps_vel_delay_buffer is not None:
            wp.launch(
                kernel=write_to_delay_buffer_vec3,
                dim=self._num_envs,
                inputs=[
                    self.measured_velocity,
                    current_time,
                    self.gps_vel_delay_buffer,
                    self.shared_time_buffer,
                    self.shared_write_index,
                    wp.int32(self.gps_vel_buffer_size)
                ],
                device=self._device
            )
            
            delayed_time = current_time - self.gps_vel_delay
            wp.launch(
                kernel=read_from_delay_buffer_vec3,
                dim=self._num_envs,
                inputs=[
                    delayed_time,
                    self.shared_time_buffer,
                    self.gps_vel_delay_buffer,
                    self.shared_write_index,
                    wp.int32(self.gps_vel_buffer_size),
                    self.measured_velocity,
                    self.gps_vel_valid
                ],
                device=self._device
            )
        
        # Attitude Sensor
        wp.launch(
            kernel=sense_attitude,
            dim=self._num_envs,
            inputs=[
                aircraft_state['orientation'],
                self.warp_seed,
                self.att_noise_std,
                self.att_enable_noise,
                current_time,
                self.att_sampling_period,
                wp.int32(self.time_step),
                self.att_last_sample_time,
                self.measured_orientation,
                self.att_valid
            ],
            device=self._device
        )
        
        # Attitude delay processing
        if self.att_delay_buffer is not None:
            wp.launch(
                kernel=write_to_delay_buffer_quat,
                dim=self._num_envs,
                inputs=[
                    self.measured_orientation,
                    current_time,
                    self.att_delay_buffer,
                    self.shared_time_buffer,
                    self.shared_write_index,
                    wp.int32(self.att_buffer_size)
                ],
                device=self._device
            )
            
            delayed_time = current_time - self.att_delay
            wp.launch(
                kernel=read_from_delay_buffer_quat,
                dim=self._num_envs,
                inputs=[
                    delayed_time,
                    self.shared_time_buffer,
                    self.att_delay_buffer,
                    self.shared_write_index,
                    wp.int32(self.att_buffer_size),
                    self.measured_orientation,
                    self.att_valid
                ],
                device=self._device
            )
        
        # Gyroscope
        wp.launch(
            kernel=sense_gyroscope,
            dim=self._num_envs,
            inputs=[
                aircraft_state['angular_vel'],
                self.warp_seed,
                self.gyro_noise_std,
                self.gyro_bias,
                self.gyro_scaling_factor,
                self.gyro_min_value,
                self.gyro_max_value,
                self.gyro_resolution,
                self.gyro_enable_noise,
                current_time,
                self.gyro_sampling_period,
                wp.int32(self.time_step),
                self.gyro_last_sample_time,
                self.measured_angular_velocity,
                self.gyro_valid
            ],
            device=self._device
        )
        
        # Gyroscope delay processing
        if self.gyro_delay_buffer is not None:
            wp.launch(
                kernel=write_to_delay_buffer_vec3,
                dim=self._num_envs,
                inputs=[
                    self.measured_angular_velocity,
                    current_time,
                    self.gyro_delay_buffer,
                    self.shared_time_buffer,
                    self.shared_write_index,
                    wp.int32(self.gyro_buffer_size)
                ],
                device=self._device
            )
            
            delayed_time = current_time - self.gyro_delay
            wp.launch(
                kernel=read_from_delay_buffer_vec3,
                dim=self._num_envs,
                inputs=[
                    delayed_time,
                    self.shared_time_buffer,
                    self.gyro_delay_buffer,
                    self.shared_write_index,
                    wp.int32(self.gyro_buffer_size),
                    self.measured_angular_velocity,
                    self.gyro_valid
                ],
                device=self._device
            )
        
        self.time_step += 1
    
    def get_sensor_measurements(self) -> dict:
        """Get current sensor measurements as dictionary"""
        return {
            'gps_position': self.measured_position,
            'gps_velocity': self.measured_velocity,
            'gyroscope': self.measured_angular_velocity,
            'attitude_sensor': self.measured_orientation,
            'validity': {
                'gps_position': self.gps_pos_valid,
                'gps_velocity': self.gps_vel_valid,
                'gyroscope': self.gyro_valid,
                'attitude_sensor': self.att_valid
            }
        }
    
    def reset(self):
        """Reset sensor system"""
        self.time_step = 0
        self.measured_position.zero_()
        self.measured_velocity.zero_()
        self.measured_orientation.zero_()
        self.measured_angular_velocity.zero_()
        self.gps_pos_last_sample_time.fill_(-1.0)
        self.gps_vel_last_sample_time.fill_(-1.0)
        self.att_last_sample_time.fill_(-1.0)
        self.gyro_last_sample_time.fill_(-1.0)
        self.gps_pos_valid.zero_()
        self.gps_vel_valid.zero_()
        self.att_valid.zero_()
        self.gyro_valid.zero_()
        
        # Reset delay buffers
        if self.gps_pos_delay_buffer is not None:
            self.gps_pos_delay_buffer.zero_()
        
        if self.gps_vel_delay_buffer is not None:
            self.gps_vel_delay_buffer.zero_()
        
        if self.att_delay_buffer is not None:
            self.att_delay_buffer.zero_()
        
        if self.gyro_delay_buffer is not None:
            self.gyro_delay_buffer.zero_()