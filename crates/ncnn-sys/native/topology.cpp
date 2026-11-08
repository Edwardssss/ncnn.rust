#include "topology.h"
#include <algorithm>
#include <climits>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#include <windows.h>
#elif defined(__linux__)
#include <cerrno>
#include <sched.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(__linux__)
static std::vector<unsigned long> thread_mask()
{
    std::vector<unsigned long> mask(sizeof(cpu_set_t) / sizeof(unsigned long), 0);
    while (sched_getaffinity(0, mask.size() * sizeof(unsigned long), (cpu_set_t*)mask.data()) != 0)
    {
        if (errno != EINVAL || mask.size() > (size_t)INT_MAX / (2 * sizeof(unsigned long) * CHAR_BIT))
            throw std::runtime_error("CPU affinity query failed");
        mask.resize(mask.size() * 2, 0);
    }
    return mask;
}
#endif

#if defined(_WIN32) || defined(__linux__)
static std::vector<uint64_t> processors(bool current)
{
    std::vector<uint64_t> ids;
#ifdef _WIN32
    if (current)
    {
        GROUP_AFFINITY affinity = {};
        if (!GetThreadGroupAffinity(GetCurrentThread(), &affinity))
            throw std::runtime_error("Thread group query failed");
        for (unsigned int bit = 0; bit < sizeof(KAFFINITY) * CHAR_BIT; bit++)
            if (affinity.Mask & ((KAFFINITY)1 << bit))
                ids.push_back(((uint64_t)affinity.Group << 32) | bit);
    }
    else
    {
        DWORD size = 0;
        GetLogicalProcessorInformationEx(RelationGroup, 0, &size);
        if (!size)
            throw std::runtime_error("Processor group query failed");
        std::vector<unsigned char> storage(size);
        if (!GetLogicalProcessorInformationEx(RelationGroup, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)storage.data(), &size))
            throw std::runtime_error("Processor group query failed");
        const size_t offset = offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Group) + offsetof(GROUP_RELATIONSHIP, GroupInfo);
        if (size < offset)
            throw std::runtime_error("Invalid processor group data");
        const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* info = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)storage.data();
        if (info->Relationship != RelationGroup || info->Group.ActiveGroupCount > (size - offset) / sizeof(PROCESSOR_GROUP_INFO))
            throw std::runtime_error("Invalid processor group data");
        for (WORD group = 0; group < info->Group.ActiveGroupCount; group++)
        {
            const KAFFINITY mask = info->Group.GroupInfo[group].ActiveProcessorMask;
            for (unsigned int bit = 0; bit < sizeof(KAFFINITY) * CHAR_BIT; bit++)
                if (mask & ((KAFFINITY)1 << bit))
                    ids.push_back(((uint64_t)group << 32) | bit);
        }
    }
#else
    if (current)
    {
        const std::vector<unsigned long> mask = thread_mask();
        for (size_t word = 0; word < mask.size(); word++)
            for (unsigned int bit = 0; bit < sizeof(unsigned long) * CHAR_BIT; bit++)
                if (mask[word] & (1ul << bit))
                    ids.push_back(word * sizeof(unsigned long) * CHAR_BIT + bit);
    }
    else
    {
        std::ifstream file("/sys/devices/system/cpu/online");
        std::string list;
        if (!std::getline(file, list))
            throw std::runtime_error("Online CPU query failed");
        size_t cursor = 0;
        while (cursor < list.size())
        {
            size_t length = 0;
            uint64_t first = std::stoull(list.substr(cursor), &length);
            cursor += length;
            uint64_t last = first;
            if (cursor < list.size() && list[cursor] == '-')
            {
                last = std::stoull(list.substr(++cursor), &length);
                cursor += length;
            }
            if (last < first || last > INT_MAX)
                throw std::runtime_error("Invalid CPU list");
            for (uint64_t id = first; id <= last; id++)
                ids.push_back(id);
            if (cursor < list.size() && list[cursor++] != ',')
                throw std::runtime_error("Invalid CPU separator");
        }
    }
#endif
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}
#endif

extern "C" int ncnn_rs_os_affinity_supported(void)
{
#if defined(_WIN32) || defined(__linux__)
    return 1;
#else
    return 0;
#endif
}

extern "C" size_t ncnn_rs_os_processors(int current, uint64_t* ids, size_t capacity)
{
#if defined(_WIN32) || defined(__linux__)
    if (current != 0 && current != 1)
        return SIZE_MAX;
    try
    {
        const std::vector<uint64_t> found = processors(current != 0);
        if (ids && capacity >= found.size())
            std::copy(found.begin(), found.end(), ids);
        return found.size();
    }
    catch (...)
    {
        return SIZE_MAX;
    }
#else
    (void)current;
    (void)ids;
    (void)capacity;
    return SIZE_MAX;
#endif
}

extern "C" int ncnn_rs_os_physical_count(void)
{
    try
    {
#ifdef _WIN32
        DWORD size = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, 0, &size);
        if (!size)
            return -1;
        std::vector<unsigned char> storage(size);
        if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)storage.data(), &size))
            return -1;
        int count = 0;
        for (size_t offset = 0; offset < size;)
        {
            if (size - offset < 2 * sizeof(DWORD))
                return -1;
            const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX* entry = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(storage.data() + offset);
            if (entry->Size < 2 * sizeof(DWORD) || entry->Size > size - offset)
                return -1;
            if (entry->Relationship == RelationProcessorCore)
                count++;
            offset += entry->Size;
        }
        return count;
#elif defined(__linux__)
        std::set<std::string> cores;
        for (uint64_t id : processors(false))
        {
            std::ifstream file("/sys/devices/system/cpu/cpu" + std::to_string(id) + "/topology/thread_siblings_list");
            std::string siblings;
            if (!std::getline(file, siblings) || siblings.empty())
                return -1;
            cores.insert(siblings);
        }
        return (int)cores.size();
#elif defined(__APPLE__)
        int count = 0;
        size_t size = sizeof(count);
        return sysctlbyname("hw.physicalcpu", &count, &size, 0, 0) == 0 ? count : -1;
#else
        return NCNN_RS_UNSUPPORTED;
#endif
    }
    catch (...)
    {
        return -1;
    }
}

extern "C" int ncnn_rs_thread_affinity_set(const uint64_t* ids, size_t count)
{
#if defined(_WIN32) || defined(__linux__)
    if (!ids || !count)
        return -1;
    try
    {
        std::vector<uint64_t> requested(ids, ids + count);
        std::sort(requested.begin(), requested.end());
        requested.erase(std::unique(requested.begin(), requested.end()), requested.end());
        const std::vector<uint64_t> online = processors(false);
        if (!std::includes(online.begin(), online.end(), requested.begin(), requested.end()))
            return -1;
#ifdef _WIN32
        GROUP_AFFINITY next = {}, previous = {};
        next.Group = (WORD)(requested[0] >> 32);
        for (uint64_t id : requested)
        {
            if ((id >> 32) != next.Group || (uint32_t)id >= sizeof(KAFFINITY) * CHAR_BIT)
                return -1;
            next.Mask |= (KAFFINITY)1 << (uint32_t)id;
        }
        if (!SetThreadGroupAffinity(GetCurrentThread(), &next, &previous))
            return -1;
        bool accepted = false;
        try
        {
            accepted = processors(true) == requested;
        }
        catch (...)
        {
        }
        if (accepted)
            return 0;
        SetThreadGroupAffinity(GetCurrentThread(), &previous, 0);
#else
        const std::vector<unsigned long> previous = thread_mask();
        const size_t bits = sizeof(unsigned long) * CHAR_BIT;
        std::vector<unsigned long> next((size_t)(requested.back() / bits + 1), 0);
        for (uint64_t id : requested)
            next[(size_t)(id / bits)] |= 1ul << (id % bits);
        if (sched_setaffinity(0, next.size() * sizeof(unsigned long), (const cpu_set_t*)next.data()) != 0)
            return -1;
        bool accepted = false;
        try
        {
            accepted = processors(true) == requested;
        }
        catch (...)
        {
        }
        if (accepted)
            return 0;
        sched_setaffinity(0, previous.size() * sizeof(unsigned long), (const cpu_set_t*)previous.data());
#endif
        return -1;
    }
    catch (...)
    {
        return -1;
    }
#else
    (void)ids;
    (void)count;
    return NCNN_RS_UNSUPPORTED;
#endif
}
