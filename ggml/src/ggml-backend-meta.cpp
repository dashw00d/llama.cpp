#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE:
            return "EXCLUSIVE";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_splite_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_splite_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types, 
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

//
// meta backend buffer
//

struct ggml_backend_meta_buffer_context {
    std::map<std::pair<const ggml_tensor *, bool>, ggml_backend_meta_split_state> split_state_cache;
    std::map<          const ggml_tensor *,        std::vector<ggml_tensor *>>    simple_tensors;

    struct buffer_config {
        ggml_context          * ctx;
        ggml_backend_buffer_t   buf;

        buffer_config(ggml_context * ctx, ggml_backend_buffer_t buf) : ctx(ctx), buf(buf) {}
    };
    std::vector<buffer_config> buf_configs;

    int debug;

    ggml_backend_meta_buffer_context() {
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (auto & [ctx, buf] : buf_ctx->buf_configs) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    delete buf_ctx;
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
    GGML_ASSERT(split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = buf_ctx->buf_configs[j].ctx;
        ggml_backend_buffer_t   simple_buf = buf_ctx->buf_configs[j].buf;

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = split_state.ne[j];
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        } else if (split_dim == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE) {
            // EXCLUSIVE: all GPUs get full-size allocation (needed for broadcast).
            // The subgraph skip logic avoids computing on non-owning GPUs.
            // This wastes memory but avoids zero-size tensor creation issues.
            // ne[] stays unchanged — same as MIRRORED for allocation purposes.
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_offs = tensor->view_offs;
        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS && t_ij->view_offs > tensor->nb[split_dim]) {
            t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
        }
        t_ij->view_src = tensor->view_src;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }
    buf_ctx->simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;

            if (n_bufs > 1) {
                // Serialize dispatch for correctness.
                // The previous threaded fan-out assumed all backend set_tensor paths were
                // host-thread-safe, but that is not guaranteed across backends/queues.
                std::vector<size_t> offsets_j(n_bufs);
                size_t off = 0;
                for (size_t j = 0; j < n_bufs; j++) {
                    offsets_j[j] = off;
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    off += simple_tensor->nb[split_state.axis + 1];
                }
                GGML_ASSERT(off == chunk_size_full);

                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                    const size_t simple_offset = i_start * chunk_size_j;
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offsets_j[j], simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                }
            } else {
                // Single GPU — no threading overhead
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
                const size_t chunk_size_0 = simple_tensor->nb[split_state.axis + 1];
                const size_t simple_offset = i_start * chunk_size_0;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data, simple_offset, chunk_size_0, i_stop - i_start, chunk_size_0, chunk_size_full);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            if (n_bufs > 1) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ggml_backend_tensor_set(simple_tensor, data, offset, size);
                }
            } else {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            const int64_t ne = ggml_nelements(tensor);
            std::vector<float> tmp;
            tmp.reserve(ne);
            for (int64_t i = 0; i < ne; i++) {
                tmp.push_back(((const float *) data)[i] / n_bufs);
            }
            if (n_bufs > 1) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ggml_backend_tensor_set(simple_tensor, tmp.data(), offset, size);
                }
            } else {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
                ggml_backend_tensor_set(simple_tensor, tmp.data(), offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE: {
            // EXCLUSIVE: set on the owning GPU only (the one with ne[j] > 0)
            for (size_t j = 0; j < n_bufs; j++) {
                if (split_state.ne[j] > 0) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ggml_backend_tensor_set(simple_tensor, data, offset, size);
                    break;
                }
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE: {
            // EXCLUSIVE: get from the owning GPU (the one with ne[j] > 0)
            for (size_t j = 0; j < n_bufs; j++) {
                if (split_state.ne[j] > 0) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ggml_backend_tensor_get(simple_tensor, data, offset, size);
                    break;
                }
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->buf_configs.size();
}

ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->buf_configs.size());
    return buf_ctx->buf_configs[index].buf;
}

struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->buf_configs.size());

    auto it = buf_ctx->simple_tensors.find(tensor);
    if (it == buf_ctx->simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context();
    size_t max_size = 0;
    buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_backend_buffer_t simple_buf = ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(simple_buf));
        buf_ctx->buf_configs.emplace_back(ggml_init(params), simple_buf);
    }

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context();
    meta_buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs.emplace_back(ggml_init(params), nullptr);
    }

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor(meta_buf, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs[i].buf = ggml_backend_alloc_ctx_tensors_from_buft(
            meta_buf_ctx->buf_configs[i].ctx, ggml_backend_meta_buft_simple_buft(buft, i));
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->buf_configs[i].buf));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config> cgraphs;
        std::vector<ggml_tensor *> nodes;
        ggml_backend_buffer_ptr    buf;

        backend_config(ggml_backend_t backend) : backend(backend) {}
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        name = "Meta(";
        backend_configs.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            backend_configs.emplace_back(ggml_backend_dev_init(simple_dev, params));
        }
        name += ")";
    }

    ~ggml_backend_meta_context() {
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }

    size_t n_reduce_steps() const {
        return std::ceil(std::log2(backend_configs.size()));
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE: {
            for (size_t j = 0; j < n_backends; j++) {
                if (split_state.ne[j] > 0) {
                    ggml_backend_tensor_set_async(
                        ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
                    break;
                }
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE: {
            for (size_t j = 0; j < n_backends; j++) {
                if (split_state.ne[j] > 0) {
                    ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
                    break;
                }
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
    }
    for (size_t j = 0; j < n_backends; j++) {
        auto & bcj = backend_ctx->backend_configs[j];

        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                bcj.nodes[i] = node;
                continue;
            }
            bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
            GGML_ASSERT(bcj.nodes[i]);

            // EP: For MUL_MAT_ID with SPLIT_AXIS_2 src0 (expert weights), encode
            // expert_offset and n_local_experts into the per-GPU simple tensor's op_params.
            // The SYCL backend reads these to know which global expert IDs this GPU owns.
            //   op_params[1] = expert_offset (first global expert index on this GPU)
            //   op_params[2] = n_local_experts (number of experts on this GPU)
            if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] != nullptr &&
                    ggml_backend_buffer_is_meta(node->src[0]->buffer)) {
                const ggml_backend_meta_split_state src0_ss = ggml_backend_meta_get_split_state(node->src[0], /*assume_sync =*/ true);
                if (src0_ss.axis == GGML_BACKEND_SPLIT_AXIS_2) {
                    // Calculate expert_offset for GPU j: sum of ne[0..j-1]
                    int32_t expert_offset = 0;
                    for (size_t k = 0; k < j; k++) {
                        expert_offset += (int32_t) src0_ss.ne[k];
                    }
                    int32_t n_local_experts = (int32_t) src0_ss.ne[j];
                    // EP op_params layout for MUL_MAT_ID:
                    //   [0] = precision (existing, usually 0)
                    //   [1] = expert_offset: first global expert index on this GPU
                    //   [2] = n_local_experts: number of experts on this GPU
                    //   [3] = EP flag (1 = EP mode, backend must zero output before compute)
                    // EP layout: [1]=expert_offset, [2]=n_local_experts
                    // [0] preserved (precision). Phase 3 reads [1] and [2].
                    // EP layout: [1]=expert_offset, [2]=n_local_experts (>0 signals EP mode)
                    bcj.nodes[i]->op_params[1] = expert_offset;
                    bcj.nodes[i]->op_params[2] = n_local_experts;
                    // DEBUG: confirm EP params are being set (first gate_exps only, all backends)
                    static bool ep_first_printed = false;
                    if (!ep_first_printed && strstr(node->src[0]->name, "ffn_gate_exps") != nullptr) {
                        fprintf(stderr, "EP META SET [j=%zu, node=%s]: expert_offset=%d n_local=%d\n",
                                j, node->src[0]->name, expert_offset, n_local_experts);
                        if (j + 1 == n_backends) ep_first_printed = true;
                    }
                } else {
                    static bool not_ep_printed = false;
                    if (!not_ep_printed) {
                        not_ep_printed = true;
                        fprintf(stderr, "EP META SKIP [j=%zu, node=%s]: src0_axis=%s (not SPLIT_AXIS_2)\n",
                                j, node->src[0]->name,
                                ggml_backend_meta_split_axis_name(src0_ss.axis));
                    }
                }
            }
        }
    }

    size_t n_subgraphs  = 0;
    size_t max_tmp_size = 0;
    {
        // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
        auto get_i_delayed = [&](const int i) -> int {
            int id = i; // i_delayed
            int idr = i; // i_delayed return, last safe return value

            ggml_tensor * node = cgraph->nodes[id];
            int32_t n_used = ggml_node_get_use_count(cgraph, id);
            if (id + 1 >= cgraph->n_nodes) {
                return idr;
            }
            {
                ggml_tensor * next = cgraph->nodes[id+1];
                if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                        ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                        ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                    node = next;
                    id++;
                    idr = id;
                    n_used = ggml_node_get_use_count(cgraph, id);
                }
            }
            if (id + 1 >= cgraph->n_nodes) {
                return idr;
            }
            {
                ggml_tensor * next = cgraph->nodes[id+1];
                if (next->op == GGML_OP_MUL && next->src[0] == node &&
                        ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                    node = next;
                    id++;
                    idr = id;
                    n_used = ggml_node_get_use_count(cgraph, id);
                }
            }

            if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                return idr;
            }
            for (int32_t k = 0; k < n_used; k++) {
                ggml_tensor * next = cgraph->nodes[id+1];
                if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                        next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                        ggml_node_get_use_count(cgraph, id+1) != 1) {
                    return idr;
                }
                id++;
            }
            {
                ggml_tensor * next = cgraph->nodes[id+1];
                if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                        next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                    return idr;
                }
                id++;
            }
            for (int32_t k = 0; k < n_used - 2; k++) {
                ggml_tensor * next = cgraph->nodes[id+1];
                if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                        next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                    return idr;
                }
                id++;
            }
            idr = id;
            return idr;
        };

        const bool debug_splits = getenv("GGML_META_DEBUG_SPLITS") != nullptr;
        size_t n_boundaries = 0;

        int i_start = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                continue;
            }
            if (node->op == GGML_OP_ADD_ID) {
                GGML_ASSERT(ggml_backend_meta_get_split_state(node->src[1], /*assume_sync =*/ false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL);
            }
            const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);

            if (debug_splits) {
                fprintf(stderr, "[META_SPLITS] node[%d] op=%-20s axis=%s\n",
                        i, ggml_op_name(node->op),
                        ggml_backend_meta_split_axis_name(split_state.axis));
            }

            // EP subgraph merging: defer AllReduce for intermediate PARTIAL nodes.
            // In an EP MoE block: gate(PARTIAL) → up(PARTIAL) → GLU(PARTIAL) → down(PARTIAL) → AllReduce.
            // We defer the AllReduce until the LAST MUL_MAT_ID in the sequence (the down projection).
            //
            // Strategy: if this node is PARTIAL, check if there's another MUL_MAT_ID ahead
            // that also operates on SPLIT_AXIS_2 expert tensors. If so, we're still in the
            // middle of the MoE block — defer. This avoids peeking at intermediate ops (GLU etc.)
            // whose split states depend on inputs that haven't been fully resolved yet.
            bool defer_ep_allreduce = false;
            if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL && i + 1 < cgraph->n_nodes) {
                // Look ahead for another MUL_MAT_ID with SPLIT_AXIS_2 src0 in this MoE block
                for (int peek = i + 1; peek < cgraph->n_nodes && peek <= i + 20; peek++) {
                    ggml_tensor * pn = cgraph->nodes[peek];
                    // Skip view nodes
                    if (pn->view_src != nullptr && pn->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(pn->view_src->buffer)) {
                        continue;
                    }
                    // If we hit another MUL_MAT_ID with SPLIT_AXIS_2 expert weights, defer
                    if (pn->op == GGML_OP_MUL_MAT_ID && pn->src[0] != nullptr) {
                        const ggml_backend_meta_split_state src0_state = ggml_backend_meta_get_split_state(pn->src[0], /*assume_sync =*/ true);
                        if (src0_state.axis == GGML_BACKEND_SPLIT_AXIS_2) {
                            defer_ep_allreduce = true;
                            break;
                        }
                    }
                    // If we hit a PARTIAL → MIRRORED boundary (non-MUL_MAT_ID PARTIAL consumer), stop
                    // Element-wise ops between MUL_MAT_IDs (GLU, MUL, etc.) are fine to skip past
                    if (pn->op == GGML_OP_MUL_MAT || pn->op == GGML_OP_ADD) {
                        // Check if this is an AllReduce-triggering ADD (residual connection)
                        // If both inputs are PARTIAL/MIRRORED mix, this is the end of the MoE block
                        break;
                    }
                }
            }

            if (debug_splits && split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                fprintf(stderr, "[META_SPLITS]   PARTIAL node[%d] defer_ep_allreduce=%s\n",
                        i, defer_ep_allreduce ? "true" : "false");
            }

            if (!defer_ep_allreduce && split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
            }
            // Split subgraph at MoE entry: when the NEXT node is a MUL_MAT that feeds
            // into MUL_MAT_ID with SPLIT_AXIS_2 (expert weights). This creates one
            // boundary per layer between DeltaNet/attention and MoE, not hundreds.
            bool exclusive_to_ep_transition = false;
            // Split at post-attention norm: the last EXCLUSIVE tensor before MoE.
            // Gate MUL_MAT moves into EP subgraph (cheap, runs on all GPUs).
            // Only attn_post_norm needs broadcast (16KB per token).
            if (node->name && strstr(node->name, "attn_post_norm") != nullptr &&
                strstr(node->name, "reshaped") == nullptr) {
                exclusive_to_ep_transition = true;
            }

            const bool new_subgraph = i + 1 == cgraph->n_nodes ||
                (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL && !defer_ep_allreduce) ||
                exclusive_to_ep_transition;
            if (!new_subgraph) {
                continue;
            }

            {
                const int i_orig = i;
                i = get_i_delayed(i);
                ggml_tensor * delayed_node = cgraph->nodes[i];
                fprintf(stderr, "EP ALLREDUCE [get_i_delayed]: tensor=%s, i=%d, delayed_i=%d, extended=%s\n",
                        delayed_node->name, i_orig, i, (i > i_orig) ? "YES" : "NO");
            }

            if (debug_splits && split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                fprintf(stderr, "[META_SPLITS] --- boundary at node[%d]: new subgraph #%zu starts at node[%d]\n",
                        i, n_subgraphs + 1, i + 1);
                n_boundaries++;
            }

            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                bcj.cgraphs[n_subgraphs].offset = i_start;
            }
            n_subgraphs++;
            i_start = i + 1;
        }
        GGML_ASSERT(i_start == cgraph->n_nodes);

        if (debug_splits) {
            fprintf(stderr, "[META_SPLITS] summary: n_subgraphs=%zu, n_boundaries=%zu\n",
                    n_subgraphs, n_boundaries);
        }
    }

    if (max_tmp_size > backend_ctx->max_tmp_size) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.buf.reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
        }
        backend_ctx->max_tmp_size = max_tmp_size;
    }


    if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
        backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
        const size_t n_reduce_steps = backend_ctx->n_reduce_steps();
        const size_t n_nodes_per_device = 2 * n_reduce_steps; // tmp + ADD per step
        const size_t n_cgraphs_per_device = n_reduce_steps;    // 1 ADD graph per step
        const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
        const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
        const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
        ggml_init_params params = {
            /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        backend_ctx->ctx.reset(ggml_init(params));
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i = 0; i < n_subgraphs; i++) {
                bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), cgraph->n_nodes, /*grads =*/ false);
            }
        }
        backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
        for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
            backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
        }
        backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
        for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
            backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
        }
    }

    for (size_t j = 0; j < n_backends; j++) {
        auto & bcj = backend_ctx->backend_configs[j];
        for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
            ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
            const size_t i_node_start = bcj.cgraphs[i_graph].offset;
            const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
            cgraph_ij->n_nodes = i_node_stop - i_node_start;
            ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
            for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                ggml_tensor * node_ij = bcj.nodes[i_node];
                cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };
    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        for (size_t offset_j = 1; offset_j < n_backends; offset_j *= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < n_backends; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other > j) {
                    continue;
                }

                auto & bcj1 = backend_ctx->backend_configs[j];
                auto & bcj2 = backend_ctx->backend_configs[j_other];

                ggml_tensor * node1 = bcj1.cgraphs[i].cgraph_main->nodes[bcj1.cgraphs[i].cgraph_main->n_nodes - 1];
                ggml_tensor * node2 = bcj2.cgraphs[i].cgraph_main->nodes[bcj2.cgraphs[i].cgraph_main->n_nodes - 1];
                GGML_ASSERT(ggml_is_contiguous(node1));
                GGML_ASSERT(ggml_is_contiguous(node2));

                // Tmp tensors to receive P2P copies
                ggml_tensor * node_tmp_1 = get_node_aux(node1);
                node_tmp_1->buffer = bcj1.buf.get();
                node_tmp_1->data = ggml_backend_buffer_get_base(bcj1.buf.get());

                ggml_tensor * node_tmp_2 = get_node_aux(node2);
                node_tmp_2->buffer = bcj2.buf.get();
                node_tmp_2->data = ggml_backend_buffer_get_base(bcj2.buf.get());

                // 2 P2P copies: exchange full buffers
                ggml_backend_tensor_copy_async(bcj1.backend, bcj2.backend, node1, node_tmp_2);
                ggml_backend_tensor_copy_async(bcj2.backend, bcj1.backend, node2, node_tmp_1);

                // Local ADD: node1 += tmp1 (in-place via view)
                ggml_tensor * node_red_1 = get_node_aux(node1);
                node_red_1->view_src = node1->view_src == nullptr ? node1 : node1->view_src;
                node_red_1->view_offs = node1->view_offs;
                node_red_1->op = GGML_OP_ADD;
                node_red_1->src[0] = node1;
                node_red_1->src[1] = node_tmp_1;
                node_red_1->flags |= GGML_TENSOR_FLAG_COMPUTE;
                ggml_backend_view_init(node_red_1);

                // Local ADD: node2 += tmp2 (in-place via view)
                ggml_tensor * node_red_2 = get_node_aux(node2);
                node_red_2->view_src = node2->view_src == nullptr ? node2 : node2->view_src;
                node_red_2->view_offs = node2->view_offs;
                node_red_2->op = GGML_OP_ADD;
                node_red_2->src[0] = node2;
                node_red_2->src[1] = node_tmp_2;
                node_red_2->flags |= GGML_TENSOR_FLAG_COMPUTE;
                ggml_backend_view_init(node_red_2);

                // Build 1-node cgraphs for the ADD ops
                ggml_cgraph * cgraph_aux_1 = get_cgraph_aux();
                cgraph_aux_1->nodes[0] = node_red_1;
                cgraph_aux_1->n_nodes = 1;
                step_cgraphs[j] = cgraph_aux_1;

                ggml_cgraph * cgraph_aux_2 = get_cgraph_aux();
                cgraph_aux_2->nodes[0] = node_red_2;
                cgraph_aux_2->n_nodes = 1;
                step_cgraphs[j_other] = cgraph_aux_2;
            }

            // Execute local ADDs for this step
            for (size_t j = 0; j < n_backends; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
        return GGML_STATUS_SUCCESS;
    };


    static int meta_call_count = 0;
    meta_call_count++;
    auto t_total_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < n_subgraphs; i++) {
        // Determine if this subgraph is EXCLUSIVE (non-MoE, one GPU only)
        // by checking the split state of the last ORIGINAL node (not per-GPU copy).
        bool subgraph_exclusive = false;
        size_t exclusive_owner = 0;

        if (n_backends > 1) {
            // Get the original graph node index for this subgraph's last node
            auto & bc0 = backend_ctx->backend_configs[0];
            size_t i_node_start = bc0.cgraphs[i].offset;
            size_t i_node_stop = (i + 1 < n_subgraphs) ? bc0.cgraphs[i + 1].offset : cgraph->n_nodes;
            if (i_node_stop > i_node_start) {
                ggml_tensor * orig_last = cgraph->nodes[i_node_stop - 1];
                if (orig_last->buffer && ggml_backend_buffer_is_meta(orig_last->buffer)) {
                    ggml_backend_meta_split_state last_state = ggml_backend_meta_get_split_state(orig_last, true);
                    if (last_state.axis == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE) {
                        subgraph_exclusive = true;
                        // Find owner by looking at WEIGHT tensors (they have ne[owner]>0, ne[others]=0).
                        // Compute tensors have ne[j]>0 for all GPUs (full allocation).
                        bool found_owner = false;
                        for (size_t ni = i_node_start; ni < i_node_stop && !found_owner; ni++) {
                            ggml_tensor * n = cgraph->nodes[ni];
                            for (int s = 0; s < GGML_MAX_SRC; s++) {
                                if (n->src[s] == nullptr) break;
                                ggml_tensor * src = n->src[s];
                                if (src->buffer && ggml_backend_buffer_is_meta(src->buffer) &&
                                    ggml_backend_buffer_get_usage(src->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                                    ggml_backend_meta_split_state ws = ggml_backend_meta_get_split_state(src, true);
                                    if (ws.axis == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE) {
                                        // Check which GPU has non-zero ne (the owner)
                                        int zero_count = 0;
                                        for (size_t j = 0; j < n_backends; j++) {
                                            if (ws.ne[j] == 0) zero_count++;
                                            else exclusive_owner = j;
                                        }
                                        if (zero_count > 0) { found_owner = true; break; }
                                    }
                                }
                            }
                        }
                        if (!found_owner) {
                            exclusive_owner = 0; // fallback
                        }
                    }
                }
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            // EXCLUSIVE subgraph: only owning GPU computes
            if (subgraph_exclusive && j != exclusive_owner) {
                continue;
            }
            auto & bcj = backend_ctx->backend_configs[j];
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        // EXCLUSIVE→EP broadcast: after an EXCLUSIVE subgraph, broadcast ALL
        // tensors that cross into the next EP subgraph. Finds cross-boundary
        // tensors by checking EP subgraph nodes' src[] for EXCLUSIVE tensors.
        if (subgraph_exclusive && n_backends > 1 && i + 1 < n_subgraphs) {
            // Collect unique EXCLUSIVE tensors consumed by the next subgraph
            auto & bc_owner = backend_ctx->backend_configs[exclusive_owner];
            size_t next_start = bc_owner.cgraphs[i + 1].offset;
            size_t next_stop = (i + 2 < n_subgraphs) ? bc_owner.cgraphs[i + 2].offset : cgraph->n_nodes;

            std::vector<size_t> bcast_node_indices; // indices into the EXCLUSIVE subgraph
            size_t excl_start = bc_owner.cgraphs[i].offset;
            size_t excl_stop = next_start;

            for (size_t ni = next_start; ni < next_stop; ni++) {
                ggml_tensor * ep_node = cgraph->nodes[ni];
                for (int s = 0; s < GGML_MAX_SRC; s++) {
                    if (ep_node->src[s] == nullptr) break;
                    // Check if this src was produced in the EXCLUSIVE subgraph
                    for (size_t ei = excl_start; ei < excl_stop; ei++) {
                        if (cgraph->nodes[ei] == ep_node->src[s]) {
                            // Found a cross-boundary tensor — check if already tagged
                            bool already = false;
                            for (size_t idx : bcast_node_indices) {
                                if (idx == ei) { already = true; break; }
                            }
                            if (!already) {
                                bcast_node_indices.push_back(ei);
                            }
                        }
                    }
                }
            }

            // Also check the ENTIRE remaining graph for residual connections
            // that reference tensors from this EXCLUSIVE subgraph
            for (size_t ni = next_stop; ni < (size_t)cgraph->n_nodes && ni < next_stop + 100; ni++) {
                ggml_tensor * node_after = cgraph->nodes[ni];
                for (int s = 0; s < GGML_MAX_SRC; s++) {
                    if (node_after->src[s] == nullptr) break;
                    for (size_t ei = excl_start; ei < excl_stop; ei++) {
                        if (cgraph->nodes[ei] == node_after->src[s]) {
                            bool already = false;
                            for (size_t idx : bcast_node_indices) {
                                if (idx == ei) { already = true; break; }
                            }
                            if (!already) {
                                bcast_node_indices.push_back(ei);
                            }
                        }
                    }
                }
            }

            // Broadcast each cross-boundary tensor
            static int bcast_log_count = 0;
            for (size_t idx : bcast_node_indices) {
                ggml_tensor * owner_tensor = bc_owner.nodes[idx];
                const size_t nbytes = ggml_nbytes(owner_tensor);
                if (nbytes == 0 || nbytes > 64*1024*1024) continue;

                std::vector<char> host_buf(nbytes);
                ggml_backend_tensor_get(owner_tensor, host_buf.data(), 0, nbytes);
                for (size_t j = 0; j < n_backends; j++) {
                    if (j == exclusive_owner) continue;
                    ggml_tensor * dst_tensor = backend_ctx->backend_configs[j].nodes[idx];
                    ggml_backend_tensor_set(dst_tensor, host_buf.data(), 0, nbytes);
                }

                if (bcast_log_count < 100) {
                    bcast_log_count++;
                    fprintf(stderr, "[META-BROADCAST] subgraph %zu: %zu bytes from GPU%zu (tensor=%s, idx=%zu)\n",
                            i, nbytes, exclusive_owner, owner_tensor->name, idx);
                }
            }
        }

        // AllReduce only for PARTIAL subgraphs (EP). EXCLUSIVE subgraphs use broadcast instead.
        if (n_backends > 1 && i < n_subgraphs - 1 && !subgraph_exclusive) {
            bool backend_allreduce_success = false;
            ggml_backend_allreduce_tensor_t allreduce_tensor = (ggml_backend_allreduce_tensor_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_ctx->backend_configs[0].backend)), "ggml_backend_allreduce_tensor");
            if (allreduce_tensor) {
                std::vector<ggml_backend_t> backends;
                backends.reserve(n_backends);
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    backends.push_back(bcj.backend);
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                {
                    ggml_cgraph * cgraph_i0 = backend_ctx->backend_configs[0].cgraphs[i].cgraph_main;
                    ggml_tensor * allreduce_node = cgraph_i0->nodes[cgraph_i0->n_nodes-1];
                    fprintf(stderr, "EP ALLREDUCE EXEC: tensor=%s\n", allreduce_node->name);
                }
                backend_allreduce_success = allreduce_tensor(backends.data(), nodes.data(), n_backends);
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
    if (meta_call_count <= 5 || meta_call_count % 20 == 0) {
        GGML_LOG_INFO("[META] #%d subgraphs=%zu total=%.1fms (%.1fms/subgraph)\n",
                meta_call_count, n_subgraphs, total_ms, total_ms / n_subgraphs);
    }

    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}

struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        // EXCLUSIVE: all EXCLUSIVE states are equal regardless of ne[] (which encodes owner GPU).
        // Two EXCLUSIVE tensors in the same op are on the same GPU (same layer context).
        if (a.axis == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE) {
            return true;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            if (a.ne[j] != b.ne[j]) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state homogeneous_src_split_state = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (homogeneous_src_split_state.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                homogeneous_src_split_state = src_split_states[i];
            } else if (!split_states_equal(src_split_states[i], homogeneous_src_split_state)) {
                const auto ax_cur = homogeneous_src_split_state.axis;
                const auto ax_new = src_split_states[i].axis;
                // EXCLUSIVE + EXCLUSIVE → keep EXCLUSIVE (same layer, same GPU)
                if (ax_cur == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE && ax_new == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE) {
                    continue;
                }
                // EXCLUSIVE + MIRRORED → keep EXCLUSIVE (most restrictive — data on one GPU)
                if ((ax_cur == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE && ax_new == GGML_BACKEND_SPLIT_AXIS_MIRRORED) ||
                    (ax_cur == GGML_BACKEND_SPLIT_AXIS_MIRRORED && ax_new == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE)) {
                    if (ax_new == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE) {
                        homogeneous_src_split_state = src_split_states[i];
                    }
                    continue;
                }
                // EXCLUSIVE + PARTIAL → keep EXCLUSIVE (after AllReduce, result is on all GPUs)
                if ((ax_cur == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE && ax_new == GGML_BACKEND_SPLIT_AXIS_PARTIAL) ||
                    (ax_cur == GGML_BACKEND_SPLIT_AXIS_PARTIAL && ax_new == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE)) {
                    if (ax_cur != GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE) {
                        homogeneous_src_split_state = src_split_states[i];
                    }
                    continue;
                }
                fprintf(stderr, "[META-GENERIC-FAIL] op=%s tensor=%s ax_cur=%d ax_new=%d\n",
                        ggml_op_name(tensor->op), tensor->name, (int)ax_cur, (int)ax_new);
                homogeneous_src_split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}};
                break;
            }
        }
        if (homogeneous_src_split_state.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            homogeneous_src_split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}};
        }
        // scalar_only: reject dimensional splits but allow EXCLUSIVE (not a split)
        if (scalar_only && homogeneous_src_split_state.axis >= 0 && homogeneous_src_split_state.axis < GGML_MAX_DIMS) {
            homogeneous_src_split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}};
        }
        GGML_ASSERT(homogeneous_src_split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return homogeneous_src_split_state;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_split_states[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_split_states[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        if (src_split_states[0].axis >= 0 && src_split_states[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_split_states[0].axis] == 1 && src_split_states[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_split_states[0];
        }
        if (src_split_states[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_split_states[0].axis == src_split_states[1].axis ||
                (src_split_states[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_split_states[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL))) {
            return src_split_states[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_split_states[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_split_states, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        if (src_split_states[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_split_states[1].axis >= 0 && src_split_states[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(ggml_get_op_params_i32(tensor, 0) != src_split_states[1].axis);
            return src_split_states[1];
        }
        if (src_split_states[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_split_states[0].axis >= 0 && src_split_states[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(ggml_get_op_params_i32(tensor, 0) != src_split_states[0].axis);
            return src_split_states[0];
        }
        return handle_generic(src_split_states, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        const auto a0 = src_split_states[0].axis;
        const auto a1 = src_split_states[1].axis;
        if (a0 == GGML_BACKEND_SPLIT_AXIS_MIRRORED && a1 == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}};
        }
        // EXCLUSIVE activation + MIRRORED weights → EXCLUSIVE (compute on owning GPU)
        if (a0 == GGML_BACKEND_SPLIT_AXIS_MIRRORED && a1 == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE) {
            return {GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE, {0}};
        }
        // EXCLUSIVE weights + EXCLUSIVE activation → EXCLUSIVE
        if (a0 == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE && a1 == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE) {
            return {GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE, {0}};
        }
        if (a0 == GGML_BACKEND_SPLIT_AXIS_1 && a1 == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_split_states[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            return ret;
        }
        if (a0 == GGML_BACKEND_SPLIT_AXIS_0 && a1 == GGML_BACKEND_SPLIT_AXIS_0) {
            for (size_t j = 0; j < n_bufs; j++) {
                GGML_ASSERT(src_split_states[0].ne[j] == src_split_states[1].ne[j]);
            }
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}};
        }
        GGML_ABORT("handle_mul_mat: unsupported split combination a0=%d a1=%d", (int)a0, (int)a1);
        return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}};
    };

    // Expert Parallelism: MUL_MAT_ID with SPLIT_AXIS_2 src0 (expert weights split on expert dimension)
    // Each GPU owns a subset of experts. Output is PARTIAL → AllReduce SUM merges results.
    auto handle_mul_mat_id = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        if (src_split_states[0].axis == GGML_BACKEND_SPLIT_AXIS_2) {
            // EP mode: src0 (expert weights) split on axis 2 (expert dimension)
            // src1 (activations): MIRRORED for gate/up (first MUL_MAT_ID in MoE block),
            //                     PARTIAL for down (after GLU combines gate+up outputs)
            // src2 (routing ids) should be MIRRORED
            GGML_ASSERT(src_split_states[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                        src_split_states[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL ||
                        src_split_states[1].axis == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE);
            GGML_ASSERT(src_split_states[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                        src_split_states[2].axis == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE);
            // Output is PARTIAL: each GPU computes full results for owned experts, zeros for others.
            // AllReduce SUM will combine them correctly (zeros + real values = real values).
            // Output is PARTIAL during compute (each GPU has partial results).
            // With assume_sync=true (used by buffer_init_tensor), return MIRRORED
            // because weights are fully synced after AllReduce.
            // With assume_sync=false (used by subgraph sweep), return PARTIAL
            // so downstream ops (GLU, MUL) propagate the split state correctly.
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}};
        }
        // Non-EP MUL_MAT_ID: fall through to regular TP handle_mul_mat
        return handle_mul_mat(src_split_states);
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        switch (src_split_states[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]));
                int64_t base_ne_in = 1;
                for (int dim = 0; dim <= src_split_states[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    const int64_t base_ne_out_next = base_ne_out *= tensor->ne[dim];
                    if (base_ne_out_next == base_ne_in) {
                        return {ggml_backend_meta_split_axis(dim), {0}};
                    }
                    if (base_ne_out_next > base_ne_in) {
                        GGML_ASSERT(dim + 1 < GGML_MAX_DIMS);
                        return {ggml_backend_meta_split_axis(dim + 1), {0}};
                    }
                    base_ne_out = base_ne_out_next;
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            case GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE: {
                return src_split_states[0];
            }
            default: {
                GGML_ABORT("fatal error");
                return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}};
            }
        }
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        const int axis = src_split_states[0].axis;
        bool only_views_of_non_split_dim = true;
        for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
            if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                only_views_of_non_split_dim = false;
                break;
            }
            if (dim == axis && tensor->ne[dim] != tensor->src[0]->ne[dim]) {
                only_views_of_non_split_dim = false;
                break;
            }
        }
        if (only_views_of_non_split_dim) {
            return src_split_states[0];
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0])) {
            return handle_reshape(src_split_states);
        }
        if (src_split_states[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_split_states[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_split_states[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        switch (src_split_states[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                return {ggml_backend_meta_split_axis(tensor->op_params[src_split_states[0].axis]), {0}};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            case GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE: {
                return src_split_states[0];
            }
            default: {
                GGML_ABORT("fatal error");
                return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}};
            }
        }
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_split_states[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_split_states[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                    src_split_states[1].axis == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE);
        // EXCLUSIVE src + MIRRORED dst is valid (compute writes to cache on owning GPU)
        if (!split_states_equal(src_split_states[0], src_split_states[2])) {
            GGML_ASSERT(
                (src_split_states[0].axis == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE &&
                 src_split_states[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) ||
                (src_split_states[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                 src_split_states[2].axis == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE));
        }
        // Return the more restrictive state
        if (src_split_states[0].axis == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE ||
            src_split_states[2].axis == GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE) {
            return {GGML_BACKEND_SPLIT_AXIS_EXCLUSIVE, {0}};
        }
        return src_split_states[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_split_states[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_split_states[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_split_states) -> ggml_backend_meta_split_state {
        GGML_ASSERT(                             src_split_states[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_split_states[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_split_states[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[4] == nullptr || src_split_states[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(tensor->src[4] == nullptr || src_split_states[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis <= GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t j = 0; j < n_bufs; j++) {
                    GGML_ASSERT(ret.ne[j] % granularity == 0);
                    ne_sum += ret.ne[j];
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_split_states(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_split_states[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}};
                continue;
            }
            // For GLU ops, resolve sources with assume_sync=false so EP MUL_MAT_ID
            // outputs return PARTIAL (not MIRRORED). This lets GLU propagate PARTIAL
            // through the MoE block. All other ops use assume_sync=true (standard).
            const bool sync_for_src = (tensor->op != GGML_OP_GLU);
            src_split_states[i] = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ sync_for_src);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_split_states);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_split_states);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_split_states);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_split_states);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_split_states);
            } break;
            case GGML_OP_MUL_MAT: {
                split_state = handle_mul_mat(src_split_states);
            } break;
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat_id(src_split_states);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET:
            case GGML_OP_CPY: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_split_states);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_split_states);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_split_states);
            } break;
            case GGML_OP_TRANSPOSE:
            case GGML_OP_GET_ROWS:
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_split_states);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_split_states);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE:
            case GGML_OP_PAD:
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_split_states);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI:
            case GGML_OP_FILL: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_split_states);
            } break;
            case GGML_OP_FLASH_ATTN_BACK:
            case GGML_OP_SSM_CONV:
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_GATED_DELTA_NET:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_split_states);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD: {
                split_state = handle_generic(src_split_states, /*scalar_only =*/ false);
            } break;
            case GGML_OP_GLU: {
                // In EP MoE, GLU's inputs come from MUL_MAT_ID with SPLIT_AXIS_2 expert weights.
                // Output should be PARTIAL → AllReduce merges gate/up activations per expert.
                // If any source is PARTIAL, return PARTIAL directly without checking ne consistency.
                // The ne values may differ for GLU inputs (gate vs up projections) but the
                // element-wise multiply doesn't care about shape — it just multiplies per-element.
                for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                    if (tensor->src[i] != nullptr && src_split_states[i].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        split_state = src_split_states[i];
                        break;
                    }
                }
                if (split_state.axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    split_state = handle_generic(src_split_states, false);
                }
            } break;
            default: {
                fprintf(stderr, "META BACKEND: unhandled op %s (%d), falling back to handle_generic\n",
                        ggml_op_name(tensor->op), (int)tensor->op);
                split_state = handle_generic(src_split_states, /*scalar_only =*/ true);
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool src_split_by_axis_found = false;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_split_states[i].axis < 0 || src_split_states[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (src_split_by_axis_found) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        GGML_ASSERT(   split_state.ne[j] * tensor->src[i]->ne[src_split_states[i].axis]
                            == src_split_states[i].ne[j] *         tensor->ne[split_state.axis]);
                    }
                } else {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        split_state.ne[j] = src_split_states[i].ne[j] * tensor->ne[split_state.axis];
                        if (split_state.ne[j] % tensor->src[i]->ne[src_split_states[i].axis] != 0) {
                            fprintf(stderr, "META SPLIT ASSERT FAIL: op=%s tensor=%s ne[%d]=%ld src[%zu]=%s src_ne[%d]=%ld split_ne[%zu]=%ld\n",
                                    ggml_op_name(tensor->op), tensor->name, split_state.axis,
                                    (long)tensor->ne[split_state.axis], i, tensor->src[i]->name,
                                    src_split_states[i].axis, (long)tensor->src[i]->ne[src_split_states[i].axis],
                                    j, (long)split_state.ne[j]);
                        }
                        GGML_ASSERT(split_state.ne[j] % tensor->src[i]->ne[src_split_states[i].axis] == 0);
                        split_state.ne[j] /= tensor->src[i]->ne[src_split_states[i].axis];
                    }
                }
                src_split_by_axis_found = true;
            }
            GGML_ASSERT(src_split_by_axis_found);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);

    if (buf_ctx->split_state_cache.find(key) == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key] = calculate_split_state();
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor->src[0], true);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                ne_info += std::to_string(buf_ctx->split_state_cache[key].ne[j]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key];
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE && ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    return ret;
}
