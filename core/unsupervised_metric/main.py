import argparse
from metrics import ReproMetrics, DRho


def main_metrics(args):
    from scipy import io
    import os

    test_path = args.input
    fused_path = args.fused
    sensor = args.sensor
    ratio = args.ratio
    out_dir = args.out_dir

    save_outputs_flag = args.save_outputs
    show_results_flag = args.show_results

    to_fuse = io.loadmat(test_path)
    outputs = io.loadmat(fused_path)['I_MS'].astype('float32')

    pan = to_fuse['I_PAN'].astype('float32')
    ms = to_fuse['I_MS_LR'].astype('float32')
    # H W B的输入形状
    r_q2n, r_q, r_sam, r_ergas = ReproMetrics(outputs, ms, pan, sensor, ratio, 32, 32)  # 光谱相关指标
    d_rho = DRho(outputs, pan, ratio)  # 空间相关指标

    if save_outputs_flag:
        io.savemat(
            out_dir + fused_path.split(os.sep)[-1].split('.')[0] + '_Coregistered_Reprojected_Metrics.mat',
            {
                'ReproQ2n': r_q2n,
                'ReproERGAS': r_ergas,
                'ReproSAM': r_sam,
                'ReproQ': r_q,
                'D_rho': d_rho,
            }
        )

    print("ReproQ2n:   %.5f \n"
              "ReproERGAS: %.5f \n"
              "ReproSAM    %.5f \n"
              "ReproQ:     %.5f \n"
              "Drho:       %.5f"
              % (r_q2n, r_ergas, r_sam, r_q, d_rho))

    if show_results_flag:
        from show_results import show
        show(ms, pan, outputs, ratio, "Outcomes")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Z-PNN',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Package for Full-Resolution quality assessment for pansharpening'
                                                 'It consists of Reprojected Metrics, trying to solve the '
                                                 'coregistration '
                                                 'problem and a new spatial no-reference metric.',
                                     epilog='''\
    Reference: 
    Full-resolution quality assessment for pansharpening
    G. Scarpa, M. Ciotola

    Authors: 
    Image Processing Research Group of University Federico II of Naples 
    ('GRIP-UNINA')
                                         '''
                                     )
    optional = parser._action_groups.pop()
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument("-i", "--input", type=str, required=True,
                               help='The path of the .mat file which contains the MS '
                                    'and PAN images. For more details, please refer '
                                    'to the GitHub documentation.')
    requiredNamed.add_argument('-f', '--fused', type=str, required=True, help='The path of pansharpened image.')

    requiredNamed.add_argument('-s', '--sensor', type=str, required=True, choices=["WV3", "WV2", 'GE1', "QB", "IKONOS"],
                               help='The sensor that has acquired the test image. Available sensors are '
                                    'WorldView-3 (WV3), WorldView-2 (WV2), GeoEye1 (GE1), QuickBird (QB), IKONOS')

    default_out_path = 'Outputs/'
    optional.add_argument("-o", "--out_dir", type=str, default=default_out_path,
                          help='The directory in which save the outcome.')
    optional.add_argument("--save_outputs", action="store_true",
                          help='Save the results in a .mat file. Please, use out_dir flag to indicate where to save.')

    optional.add_argument("--show_results", action="store_true", help="Enable the visualization of the outcomes.")
    optional.add_argument("--ratio", type=int, default=4, help='PAN-MS resolution ratio.')

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    main_metrics(arguments)
