import os
import argparse
import pandas as pd
import numpy as np
import snf
import sklearn.preprocessing as sp
from sklearn.feature_selection import SelectKBest, chi2
from utils_map.similarity import calculate_feature_similarity
from utils_map.map import omicsMap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', 
                        help="Input your omics data sequentially")
    parser.add_argument('--label', 
                        help="Input your sample label")
    parser.add_argument('--type', nargs='+',  
                        help="Input your omics type sequentially")
    parser.add_argument('--scale', default="minmax", choices=["minmax", "z_score"],
                        help="Choose a normalization method")
    parser.add_argument('--drm', default="none", choices=["none", "fs"],
                        help="Choose a dimensionality reduction method")
    parser.add_argument('--fs_num', nargs='+', 
                        help="Input omics feature number respectively when feature selection")
    parser.add_argument('--sm', default="cosine", choices=["cosine", "euclidean"],
                        help="Choose a method for calculating feature similarity")
    parser.add_argument('--fem', default="umap", choices=["umap", "tsne"],
                        help="Choose a method for feature embedding")
    parser.add_argument('--splite_channel', type=int,  default=0,
                        help="Choose whether to splite channel for multi-omics data")
    args = parser.parse_args()
    
    path = os.getcwd()
    model_output = path + '/results_preprocessing'
    os.makedirs(model_output)
    
    print("   [IE-MOIF] --Info: Start data standardization")
    omics_df_num = len(args.data)
    omics_df_list = []
    for i in range(omics_df_num):
        single_omics = pd.read_csv(args.data[i], header=0, index_col=0)
        single_omics = pd.DataFrame(np.nan_to_num(single_omics.values,copy=False).astype(float), index=single_omics.index.values, columns=single_omics.columns.values)
        if args.scale == "minmax":
            scaler_method = sp.MinMaxScaler()
            init_om_data =  scaler_method.fit_transform(single_omics.values)
            init_om_data_df = pd.DataFrame(init_om_data, index=single_omics.index.values, columns=single_omics.columns.values) 
        elif args.scale == "z_score":
            init_om_data = sp.scale(single_omics.values)
            init_om_data_df = pd.DataFrame(init_om_data, index=single_omics.index.values, columns=single_omics.columns.values) 

        omics_df_list.append(init_om_data_df)
    init_mo_df = pd.concat(omics_df_list,axis=1)
    init_mo_df.to_csv("./results_preprocessing/1.%s_Multi-Omics_Data.csv"%args.scale)
    label_df = pd.read_csv(args.label, header=0, index_col=0)
    np.save("./results_preprocessing/5.Data_label.npy", label_df.loc[:,"Label"].values.reshape(-1,1))
    print("   [IE-MOIF] --Info: Data prepared") 



    print("   [IE-MOIF] --Info: Start sample similarity network fusion")
    affinity_networks = snf.make_affinity(omics_df_list, metric='euclidean', K=20, mu=0.5)
    fused_network_df = pd.DataFrame(snf.snf(affinity_networks, K=20), index=omics_df_list[0].index.values, columns=omics_df_list[0].index.values) 
    print("   [IE-MOIF] --Info: Try to save fused similarity matrix...")
    fused_network_df.to_csv("./results_preprocessing/2.Sample_similarity_network_matirx.csv")
    print("   [IE-MOIF] --Info: SNF Finished")


    if args.drm == "fs":
        print("   [IE-MOIF] --Info: Start Feature Selection")
        fs_mo_df_list = []
        for i in range(omics_df_num):
            fs_num = int(args.fs_num[i])
            fs_model = SelectKBest(chi2, k=fs_num)
            fs_model.fit(omics_df_list[i].values, label_df.iloc[:,0].values.reshape(-1,1))
            feature_index = [i for i,x in enumerate(fs_model.get_support()) if x]
            fs_mo_df_list.append(omics_df_list[i].iloc[:,feature_index])
            omics_df_list[i].iloc[:,feature_index].to_csv("./results_preprocessing/3.%s_feature_selection.csv"%args.type[i])
        drm_df = pd.concat(fs_mo_df_list,axis=1)
        drm_df.to_csv("./results_preprocessing/3.FS_dimension_reduced_matirx.csv")
        print("   [IE-MOIF] --Info: FS Finished")
    else:
        drm_df = init_mo_df
        pass

    print("   [IE-MOIF] --Info: Start fusing the sample matrix with the sample similarity network")
    snf_fused_mo_df = pd.DataFrame(np.matmul(fused_network_df.values, drm_df.values), index=drm_df.index.values, columns=drm_df.columns.values)
    print("   [IE-MOIF] --Info: Try to save the sample multi-omics matrix fused with sample-similarity network...")
    snf_fused_mo_df.to_csv("./results_preprocessing/4.Multi-Omics_Data(Information Enhanced).csv")
    print("   [IE-MOIF] --Info: Sample matrix fusion Finished")

    print("   [IE-MOIF] --Info: Start calculating the feature similarity network")
    if args.sm == "cosine":
        similarity_df = calculate_feature_similarity.calculate_cosine_similarity(drm_df)
    elif args.sm == "euclidean":
        similarity_df = calculate_feature_similarity.calculate_euclidean_distances(drm_df)


    mo_type = []
    for i in args.type:
        mo_type.append(i)

    color_list = ["#96ceb4","#ffad60","#ffeead","#f6c2c2","#ee4c58","#2E8B57"]

    omics_colormaps = {}
    for i in range(omics_df_num):
        omics_colormaps[mo_type[i]] = color_list[i] 

    mo_subtypes = []
    if args.drm == "fs":
        for i in range(omics_df_num):
            for j in range(len(fs_mo_df_list[i].columns)):
                mo_subtypes.append(mo_type[i])
        IDs = drm_df.columns.values
        omics_info = {"IDs": IDs, "Subtypes": mo_subtypes}
        omics_info_df = pd.DataFrame(omics_info)
        omics_info_df.to_csv('./results_map/5.omics_color.csv', index=None)
    else:
        for i in range(omics_df_num):
            for j in range(len(omics_df_list[i].columns)):
                mo_subtypes.append(mo_type[i])
        IDs = init_mo_df.columns.values
        omics_info = {"IDs": IDs, "Subtypes": mo_subtypes}
        omics_info_df = pd.DataFrame(omics_info)
        omics_info_df.to_csv('./results_map/5.omics_color.csv', index=None)
    


    mp = omicsMap( 
                fmap_type = "grid", 
                fmap_shape = None, 
                similarity_df = similarity_df,
                split_channels = True if args.splite_channel else False,
                omics_colormaps = omics_colormaps,
                metric = args.sm,
                )

    mp.fit(method=args.fem, verbose=1, metric=args.sm)
    mp.plot_grid()
    mp.plot_scatter()

    MoInter_Transformed_Data = mp.batch_transform(snf_fused_mo_df.values)
    np.save("./results_map/5.IE-MOIF_Transformed_Data_%s.npy"%args.splite_channel, MoInter_Transformed_Data)
    print("   [IE-MOIF] --Info: Saved Integration data to './results_preprocessing/5.MoInter_Transformed_Data.npy'")

    mp.save("./results_map/IE-MOIF.mp") 