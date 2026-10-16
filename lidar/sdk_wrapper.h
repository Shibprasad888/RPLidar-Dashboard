#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void* create_driver();
int destroy_driver(void* handle);
int connect_driver(void* handle, const char* device, int baudrate);
int disconnect_driver(void* handle);
int is_connected(void* handle);
int get_device_info(void* handle, int* model, int* firmware_version, int* hardware_version, char* serial_number, int serial_number_size);
int get_health(void* handle, int* status, int* error_code);
int set_motor_speed(void* handle, unsigned short speed);
int start_scan(void* handle, bool force);
int stop_scan(void* handle);
int grab_scan_data(void* handle, unsigned short* angle_q14, unsigned int* dist_q2, unsigned char* quality, unsigned char* flag, size_t* count, unsigned int timeout_ms);

#ifdef __cplusplus
}
#endif
