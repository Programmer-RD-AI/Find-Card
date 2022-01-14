from Model import *


class Param_Tunning:
    def __init__(self, ) -> None:
        f"""
        initialize the Class
        params - dict like
        """

    def tune(self, params: dict) -> dict:
        """
        Tune all of the parameters
        """
        final_metrics = []
        model = Model()
        params = ParameterGrid(params)
        params_iter = tqdm(params)
        for param in params_iter:
            try:
                torch.cuda.empty_cache()
                model.remove_files_in_output()
                torch.cuda.empty_cache()
                params_iter.set_description(str(param))
                torch.cuda.empty_cache()
                model = Model(
                    ims_per_batch=param["IMS_PER_BATCH"],
                    batch_size_per_image=param["BATCH_SIZE_PER_IMAGE"],
                    model="COCO-Detection/" + param["MODEL"],
                    base_lr=param["BASE_LR"],
                    name=str(param),
                )
                torch.cuda.empty_cache()
                metrics = model.train()
                torch.cuda.empty_cache()
                final_metrics.append(metrics)
            except Exception as e:
                torch.cuda.empty_cache()
        return final_metrics

    def ray_tune_func(self, config):
        """
        https://docs.ray.io/en/latest/tune/index.html
        """
        base_lr = config["BASE_LR"]
        ims_per_batch = (config["IMS_PER_BATCH"], )
        batch_size_per_image = config["BATCH_SIZE_PER_IMAGE"]
        model = "COCO-Detection/" + config["MODEL"]
        model = Model(
            base_lr=base_lr,
            model=model,
            ims_per_batch=ims_per_batch,
            batch_size_per_image=batch_size_per_image,
        )
        model.remove_files_in_output()
        metrics = model.train()
        ap = metrics["metrics_coco"]["bbox.AP"]
        tune.report(average_precisions=ap)

    def ray_tune(self):
        """
        https://docs.ray.io/en/latest/tune/user-guide.html
        """
        analysis = tune.run(self.ray_tune_func,
                            config=params,
                            resources_per_trial={
                                "gpu": 0,
                                "cpu": 1
                            })
        analysis.get_best_results(metrics="average_precisions", model="max")
        df = analysis.results_df
        df.to_csv("./Logs.csv")
