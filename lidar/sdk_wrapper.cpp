#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "sl_lidar.h"
#include "sl_lidar_driver.h"

using namespace sl;

struct DriverContext {
    ILidarDriver* drv;
    IChannel* channel;
};

extern "C" {

void* create_driver()
{
    auto res = createLidarDriver();
    if (!res) {
        return nullptr;
    }

    DriverContext* ctx = new DriverContext();
    ctx->drv = *res;
    ctx->channel = nullptr;
    return ctx;
}

int destroy_driver(void* handle)
{
    if (!handle) {
        return -1;
    }
    DriverContext* ctx = static_cast<DriverContext*>(handle);
    if (ctx->drv) {
        delete ctx->drv;
        ctx->drv = nullptr;
    }
    if (ctx->channel) {
        delete ctx->channel;
        ctx->channel = nullptr;
    }
    delete ctx;
    return 0;
}

int connect_driver(void* handle, const char* device, int baudrate)
{
    if (!handle || !device) {
        return -1;
    }
    DriverContext* ctx = static_cast<DriverContext*>(handle);
    if (!ctx->drv) {
        return -1;
    }

    if (ctx->channel) {
        delete ctx->channel;
        ctx->channel = nullptr;
    }

    auto channel_res = createSerialPortChannel(std::string(device), baudrate);
    if (!channel_res) {
        return static_cast<int>(channel_res.err);
    }

    ctx->channel = *channel_res;
    if (!ctx->channel) {
        return -1;
    }

    sl_result result = ctx->drv->connect(ctx->channel);
    if (!SL_IS_OK(result)) {
        delete ctx->channel;
        ctx->channel = nullptr;
    }
    return static_cast<int>(result);
}

int disconnect_driver(void* handle)
{
    if (!handle) {
        return -1;
    }
    DriverContext* ctx = static_cast<DriverContext*>(handle);
    if (!ctx->drv) {
        return -1;
    }
    ctx->drv->disconnect();
    return 0;
}

int is_connected(void* handle)
{
    if (!handle) {
        return 0;
    }
    DriverContext* ctx = static_cast<DriverContext*>(handle);
    if (!ctx->drv) {
        return 0;
    }
    return ctx->drv->isConnected() ? 1 : 0;
}

int get_device_info(void* handle, int* model, int* firmware_version, int* hardware_version, char* serial_number, int serial_number_size)
{
    if (!handle || !model || !firmware_version || !hardware_version || !serial_number || serial_number_size <= 0) {
        return -1;
    }
    DriverContext* ctx = static_cast<DriverContext*>(handle);
    if (!ctx->drv) {
        return -1;
    }

    sl_lidar_response_device_info_t info;
    sl_result result = ctx->drv->getDeviceInfo(info);
    if (!SL_IS_OK(result)) {
        return static_cast<int>(result);
    }

    *model = info.model;
    *firmware_version = info.firmware_version;
    *hardware_version = info.hardware_version;
    int copy_size = std::min(serial_number_size - 1, static_cast<int>(sizeof(info.serialnum)));
    std::memcpy(serial_number, info.serialnum, copy_size);
    serial_number[copy_size] = '\0';
    return static_cast<int>(result);
}

int get_health(void* handle, int* status, int* error_code)
{
    if (!handle || !status || !error_code) {
        return -1;
    }
    DriverContext* ctx = static_cast<DriverContext*>(handle);
    if (!ctx->drv) {
        return -1;
    }

    sl_lidar_response_device_health_t health;
    sl_result result = ctx->drv->getHealth(health);
    if (!SL_IS_OK(result)) {
        return static_cast<int>(result);
    }

    *status = health.status;
    *error_code = health.error_code;
    return static_cast<int>(result);
}

int set_motor_speed(void* handle, unsigned short speed)
{
    if (!handle) {
        return -1;
    }
    DriverContext* ctx = static_cast<DriverContext*>(handle);
    if (!ctx->drv) {
        return -1;
    }

    sl_result result = ctx->drv->setMotorSpeed(speed);
    return static_cast<int>(result);
}

int start_scan(void* handle, bool force)
{
    if (!handle) {
        return -1;
    }
    DriverContext* ctx = static_cast<DriverContext*>(handle);
    if (!ctx->drv) {
        return -1;
    }

    sl_result result = ctx->drv->startScan(force, true);
    return static_cast<int>(result);
}

int stop_scan(void* handle)
{
    if (!handle) {
        return -1;
    }
    DriverContext* ctx = static_cast<DriverContext*>(handle);
    if (!ctx->drv) {
        return -1;
    }

    sl_result result = ctx->drv->stop();
    return static_cast<int>(result);
}

int grab_scan_data(void* handle, unsigned short* angle_q14, unsigned int* dist_q2, unsigned char* quality, unsigned char* flag, size_t* count, unsigned int timeout_ms)
{
    if (!handle || !angle_q14 || !dist_q2 || !quality || !flag || !count) {
        return -1;
    }
    DriverContext* ctx = static_cast<DriverContext*>(handle);
    if (!ctx->drv) {
        return -1;
    }

    size_t max_count = *count;
    if (max_count == 0) {
        return -1;
    }

    const size_t buffer_count = 8192;
    size_t actual_count = std::min(max_count, buffer_count);
    std::vector<sl_lidar_response_measurement_node_hq_t> nodes(actual_count);
    sl_result result = ctx->drv->grabScanDataHq(nodes.data(), actual_count, timeout_ms);
    if (!SL_IS_OK(result)) {
        *count = 0;
        return static_cast<int>(result);
    }

    ctx->drv->ascendScanData(nodes.data(), actual_count);

    for (size_t i = 0; i < actual_count; ++i) {
        angle_q14[i] = nodes[i].angle_z_q14;
        dist_q2[i] = nodes[i].dist_mm_q2;
        quality[i] = nodes[i].quality;
        flag[i] = nodes[i].flag;
    }

    *count = actual_count;
    return static_cast<int>(result);
}

}
