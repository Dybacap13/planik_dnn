#!/usr/bin/env python

import os
import yaml
import argparse
import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="choose input npy folder", type=str)
    parser.add_argument("--norm", help="choose normalization type: 0 for standardization, 1 for min-max normalization, "
                                        "2 for max-absolute, 3 por robusy scaling, other for none")
    args = parser.parse_args()

    with open(f'data/pykin/{args.file}/data_stats.yaml') as f:
        stats = yaml.safe_load(f)

    for dirs, subdirs, files in os.walk(f'data/pykin/{args.file}'):
        for file in files:
            path = os.path.join(dirs, file)

            if path.endswith('.npy'):
                # load split files
                np_file = np.load(path)
                df = pd.DataFrame(np_file)

                # de-normalize
                if 'x_' in file:
                    if stats['norm'] == 'std':
                        df = df * stats['df_std_in'] + stats['df_mean_in']
                    elif stats['norm'] == 'norm':
                        df = df * (np.array(stats['df_max_in']) - np.array(stats['df_min_in'])) + stats['df_min_in']
                    elif stats['norm'] == 'max-abs':
                        df = df * stats['df_maxabs_in']
                    elif stats['norm'] == 'iqr':
                        df = df * (np.array(stats['df_quantile75_in']) - np.array(stats['df_quantile25_in'])) + stats['df_median_in']
                
                elif 'y_' in file:
                    if stats['norm'] == 'std':
                        df = df * stats['df_std_out'] + stats['df_mean_out']
                    elif stats['norm'] == 'norm':
                        df = df * (np.array(stats['df_max_out']) - np.array(stats['df_min_out'])) + stats['df_min_out']
                    elif stats['norm'] == 'max-abs':
                        df = df * stats['df_maxabs_out']
                    elif stats['norm'] == 'iqr':
                        df = df * (np.array(stats['df_quantile75_out']) - np.array(stats['df_quantile25_out'])) + stats['df_median_out']

                # new normalization
                if 'x_' in file:
                    if args.norm == '0':
                        df = (df - stats['df_mean_in']) / stats['df_std_in']
                    elif args.norm == '1':
                        df = (df - stats['df_min_in']) / (np.array(stats['df_max_in']) - np.array(stats['df_min_in']))
                    elif args.norm == '2':
                        df = df / stats['df_maxabs_in']
                    elif args.norm == '3':
                        df = (df -  stats['df_median_in']) / (np.array(stats['df_quantile75_in']) - np.array(stats['df_quantile25_in']))
                
                elif 'y_' in file:
                    if args.norm == '0':
                        df = (df - stats['df_mean_out']) / stats['df_std_out']
                    elif args.norm == '1':
                        df = (df - stats['df_min_out']) / (np.array(stats['df_max_out']) - np.array(stats['df_min_out']))
                    elif args.norm == '2':
                        df = df / stats['df_maxabs_out']
                    elif args.norm == '3':
                        df = (df -  stats['df_median_out']) / (np.array(stats['df_quantile75_out']) - np.array(stats['df_quantile25_out']))

                # save npy
                with open(path, 'wb') as f:
                    np.save(f, df.to_numpy())

    old_norm = stats['norm']

    # save stats
    if args.norm == '0':
        stats['norm'] = 'std'
    elif args.norm == '1':
        stats['norm'] = 'norm'
    elif args.norm == '2':
        stats['norm'] = 'max-abs'
    elif args.norm == '3':
        stats['norm'] = 'iqr'
    else:
        stats['norm'] = 'none'

    print(f"Changed the normalization of {args.file} from {old_norm} to {stats['norm']}")

    with open(f'data/pykin/{args.file}/data_stats.yaml', 'w') as f:
        yaml.dump(stats, f)
