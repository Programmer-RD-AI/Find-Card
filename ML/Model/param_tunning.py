from Model import *


class Param_Tunning:
    """sumary_line

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    @staticmethod
    def tune(params: dict, ) -> dict:
        """
        Tune all of the parameters
        """
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        information_dict = {"Name": [], "AP": []}
        torch.cuda.empty_cache()
        params = ParameterGrid(params)
        torch.cuda.empty_cache()
        params_iter = tqdm(params)
        torch.cuda.empty_cache()
        for param in params_iter:
            try:
                torch.cuda.empty_cache()
                name = f'{param["models"]}-{param["batch_size_per_images"]}-{param["ims_per_batchs"]}-{param["base_lrs"]}'
                torch.cuda.empty_cache()
                params_iter.set_description(str(name))
                torch.cuda.empty_cache()
                model = Detectron2(
                    ims_per_batch=param["ims_per_batchs"],
                    batch_size_per_image=param["batch_size_per_images"],
                    model="COCO-Detection/" + param["models"],
                    base_lr=param["base_lrs"],
                    name=str(name),
                    max_iter=param["max_iters"],
                )
                torch.cuda.empty_cache()
                metrics = model.train("dangerous-animals-detection-bitbybit",
                                      param)
                torch.cuda.empty_cache()
                information_dict["Name"].append(name)
                information_dict["AP"].append(
                    list(metrics["metrics_coco"].items())[0][1]["AP"])
                torch.cuda.empty_cache()
            except:
                continue
        print(information_dict)
        information_df = pd.DataFrame(information_dict)
        information_df.to_csv("./data.csv")
        return information_dict

    @staticmethod
    def ray_tune_func(config, detectron2=True):
        """
        https://docs.ray.io/en/latest/tune/index.html
        """
        print(config)
        base_lr = config["base_lrs"]
        ims_per_batch = (config["ims_per_batchs"], )
        batch_size_per_image = config["batch_size_per_images"]
        model = "COCO-Detection/" + config["models"]
        name = f"{config['models']}-{batch_size_per_image}-{ims_per_batch}-{base_lr}"
        if detectron2:
            model = Detectron2(
                base_lr=base_lr,
                model=model,
                ims_per_batch=ims_per_batch,
                batch_size_per_image=batch_size_per_image,
                name=str(name),
            )
        model.remove_files_in_output()
        metrics = model.train("dangerous-animals-detection-bitbybit")
        ap = metrics["metrics_coco"]["bbox.AP"]
        return tune.report(average_precisions=ap)

    def ray_tune(self, params):
        """
        https://docs.ray.io/en/latest/tune/user-guide.html
        """
        ray.init()
        analysis = tune.run(self.ray_tune_func,
                            config=params,
                            resources_per_trial={
                                "cpu": 0,
                                "gpu": 1
                            })
        analysis.get_best_results(metrics="average_precisions", model="max")
        df = analysis.results_df
        df.to_csv("./Logs.csv")
