from src.utae import utae, unet3d, convlstm, convgru, fpn


def get_model(config, mode="semantic"):
    if mode == "semantic":
        if config.model == "utae":
            model = utae.UTAE(
                input_dim=config.num_bands + (1 if getattr(config, 'use_sensor_flag', False) else 0),
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
                add_time_channel=getattr(config, 'add_time_channel', False),
                time_encoding_method=getattr(config, 'time_encoding_method', 'replicate'),
                add_doy_channel=getattr(config, 'add_doy_channel', False),
                use_multi_head=getattr(config, 'use_multi_head', False),
                num_coarse_classes=getattr(config, 'num_coarse_classes', 6),
                num_grassland_classes=getattr(config, 'num_grassland_classes', 8),
                num_agriculture_classes=getattr(config, 'num_agriculture_classes', 37),
                T=10000 if getattr(config, 'use_gdd_pe', False) else 1000,
            )
        elif config.model == "unet3d":
            model = unet3d.UNet3D(
                in_channel=config.num_bands, n_classes=config.num_classes, pad_value=config.pad_value
            )
        elif config.model == "fpn":
            model = fpn.FPNConvLSTM(
                input_dim=config.num_bands,
                num_classes=config.num_classes,
                inconv=[32, 64],
                n_levels=4,
                n_channels=64,
                hidden_size=88,
                input_shape=(128, 128),
                mid_conv=True,
                pad_value=config.pad_value,
            )
        elif config.model == "convlstm":
            model = convlstm.ConvLSTM_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=config.num_bands,
                kernel_size=(3, 3),
                hidden_dim=160,
            )
        elif config.model == "convgru":
            model = convgru.ConvGRU_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=config.num_bands,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
        elif config.model == "uconvlstm":
            model = utae.RecUNet(
                input_dim=config.num_bands,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=64,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        elif config.model == "buconvlstm":
            model = utae.RecUNet(
                input_dim=config.num_bands,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=30,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        return model
    elif mode == "panoptic":
        if config.backbone == "utae":
            model = utae.UTAE(
                input_dim=config.num_bands,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=True,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        elif config.backbone == "uconvlstm":
            model = utae.RecUNet(
                input_dim=config.num_bands,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=64,
                encoder=True,
                padding_mode="zeros",
                pad_value=0,
            )
        else:
            raise NotImplementedError

        model = paps.PaPs(
            encoder=model,
            num_classes=config.num_classes,
            shape_size=config.shape_size,
            mask_conv=config.mask_conv,
            min_confidence=config.min_confidence,
            min_remain=config.min_remain,
            mask_threshold=config.mask_threshold,
        )
        return model
    else:
        raise NotImplementedError
