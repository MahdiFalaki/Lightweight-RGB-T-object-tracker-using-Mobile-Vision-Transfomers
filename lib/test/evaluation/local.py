from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.check_dir = '/models'
    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/got10k_lmdb'
    settings.got10k_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/itb'
    settings.lasot_extension_subset_path_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/lasot_lmdb'
    settings.lasot_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/lasot'
    settings.network_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/nfs'
    settings.otb_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/otb'
    settings.prj_dir = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track'
    settings.result_plot_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/output/test/result_plots'
    settings.results_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/output'
    settings.segmentation_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/output/test/segmentation_results'
    settings.tc128_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/trackingnet'
    settings.uav_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/uav'
    settings.vot18_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/vot2018'
    settings.vot22_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/vot2022'
    settings.vot_path = '/home/mark/Codes/mahdi_codes_folder/mmMobileViT_Track/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

