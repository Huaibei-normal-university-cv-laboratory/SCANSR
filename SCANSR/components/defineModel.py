def defineG(config):
    script_name = "components." + config["module_script_name"]
    print('model name: %s' % config["module_script_name"])
    class_name = config["class_name"]
    package = __import__(script_name, fromlist=True)
    network_class = getattr(package, class_name)

    model_name = config["module_script_name"]
    if model_name == 'SCANSR':
        model = network_class(upscale=config['module_params']['upsampling'])
        return model
    #  消融实验
    elif model_name == 'SCANSR_k1':
        model = network_class(upscale=config['module_params']['upsampling'])
        return model
    elif model_name == 'SCANSR_k2':
        model = network_class(upscale=config['module_params']['upsampling'])
        return model
    elif model_name == 'SCANSR_k3':
        model = network_class(upscale=config['module_params']['upsampling'])
        return model
    elif model_name == 'SCANSR_k4':
        model = network_class(upscale=config['module_params']['upsampling'])
        return model
    elif model_name == 'SCANSR_k5':
        model = network_class(upscale=config['module_params']['upsampling'])
        return model
    elif model_name == 'SCANSR_k6':
        model = network_class(upscale=config['module_params']['upsampling'])
        return model
    elif model_name == 'SCANSR_k7':
        model = network_class(upscale=config['module_params']['upsampling'])
        return model
    elif model_name == 'SCANSR_k8':
        model = network_class(upscale=config['module_params']['upsampling'])
        return model
    elif model_name == 'SCANSR_FFN':
        model = network_class(upscale=config['module_params']['upsampling'])
        return model
    elif model_name == 'SCANSR_MSFM':
        model = network_class(upscale=config['module_params']['upsampling'])
        return model

