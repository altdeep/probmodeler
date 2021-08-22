
class Experiment(pl.LightningModule): 
    def __init__(self, model): 
        super().__init__() 
        self.flow_model = model 
        self.model = flow_model.model 

    def configure_optimizers(self):
        glasses_params = self.flow_model.glasses_flow_components.parameters()
        x_params = self.flow_model.trans_modules.parameters() 

        optimizer =  torch.optim.Adam([
            {'params': x_params, 'lr': 1e-3},
            {'params': glasses_params, 'lr': 1e-3},
        ], lr=1e-3, eps=1e-5) 
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
    
    def _loss(self, **batch): 
        cond_model = pyro.condition(self.flow_model.sample, data=batch) 
        model_trace = pyro.poutine.trace(cond_model).get_trace(batch["x"].shape[0]) 
        model_trace.compute_log_prob() 

        log_probs = {}
        nats_per_dim = {}
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and site["is_observed"]:
                log_probs[name] = site["log_prob"].mean()
                log_prob_shape = site["log_prob"].shape
                value_shape = site["value"].shape
                if len(log_prob_shape) < len(value_shape):
                    dims = np.prod(value_shape[len(log_prob_shape):])
                else:
                    dims = 1.
                nats_per_dim[name] = -site["log_prob"].mean() / dims
                if self.hparams.validate:
                    print(f'at site {name} with dim {dims} and nats: {nats_per_dim[name]} and logprob: {log_probs[name]}')
                    if torch.any(torch.isnan(nats_per_dim[name])):
                        raise ValueError('got nan')

        return log_probs, nats_per_dim 


    def prep_batch(self, batch):
        x = batch[0].float() 
        context = batch[1][:,2] 
        return {"x": x, "glasses": context} 


    def training_step(self, batch, *args): 
        batch = self.prep_batch(batch) 
        log_probs, nats_per_dim = self._loss(**batch) 
        loss = torch.stack(tuple(nats_per_dim.values())).sum() 

        return loss 
