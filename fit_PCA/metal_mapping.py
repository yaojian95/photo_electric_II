# Mapping of pkl block indices to metal types and thicknesses
# Directory: E:\multi_source_info\data_dir\20260611_metal_sheet_yanjiuyuan\160kV_4mA\pixel_values

# 结构说明:
# metal_name: {
#     'indices': pkl对应的序号列表,
#     'layers': 对应的层数,
#     'unit_thickness': 单片厚度(mm),
#     'thicknesses': 实际物理厚度列表(mm)
# }

METAL_MAPPING = {
    'Zn': {
        'indices': [0, 1, 2, 3, 4],
        'layers': [5, 4, 3, 2, 1],
        'unit_thickness': 0.2,
        'thicknesses': [1.0, 0.8, 0.6, 0.4, 0.2]
    },
    'Ni': {
        'indices': [5, 6, 7, 8, 9],
        'layers': [5, 4, 3, 2, 1],
        'unit_thickness': 0.2,
        'thicknesses': [1.0, 0.8, 0.6, 0.4, 0.2]
    },
    'Sn': {
        'indices': [10, 11, 12, 13, 14],
        'layers': [5, 4, 3, 2, 1],
        'unit_thickness': 0.1,
        'thicknesses': [0.5, 0.4, 0.3, 0.2, 0.1]
    },
    'Fe': {
        'indices': [15, 16, 17, 18, 19],
        'layers': [5, 4, 3, 2, 1],
        'unit_thickness': 0.3,
        'thicknesses': [1.5, 1.2, 0.9, 0.6, 0.3]
    },
    'Ti': {
        'indices': [20, 21, 22],
        'layers': [4, 3, 2],
        'unit_thickness': 0.3,
        'thicknesses': [1.2, 0.9, 0.6]
    },
    'Pb': {
        'indices': [23, 24, 25, 26, 27],
        'layers': [5, 4, 3, 2, 1],
        'unit_thickness': 0.1,
        'thicknesses': [0.5, 0.4, 0.3, 0.2, 0.1]
    },
    'Other': {
        'indices': [28, 29, 30],
        'layers': [None, None, None],
        'unit_thickness': None,
        'thicknesses': [None, None, None]
    },
    'Cu': {
        'indices': [31, 32, 33, 34, 35],
        'layers': [5, 4, 3, 2, 1],
        'unit_thickness': 0.2,
        'thicknesses': [1.0, 0.8, 0.6, 0.4, 0.2]
    },
    'W': {
        'indices': [36, 37, 38, 39, 40],
        'layers': [5, 4, 3, 2, 1],
        'unit_thickness': 0.05,
        'thicknesses': [0.25, 0.20, 0.15, 0.10, 0.05]
    }
}
