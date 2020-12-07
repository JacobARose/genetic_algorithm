
from genetic_algorithm.generation.generation import Generation
from datetime import datetime


# CUDA_VISIBLE_DEVICES=i wandb agent xxxx

def main(data, config, class_encoder=None, best_organism = None, verbose=True, debug=False):
    print(config)
    start_time = datetime.now().ctime()
    print(f'Start time = {start_time}')
    
    for phase in range(config.generation.num_phases):
        print("PHASE {}".format(phase))
#         try:
        if True:
            generation = Generation(data=data,
                                    generation_config=config['generation'],
                                    organism_config=config['organism'],
                                    phase=phase,
                                    previous_best_organism=best_organism,
                                    class_encoder=class_encoder,
                                    verbose=verbose,
                                    debug=debug)

            best_organism = generation.run_phase()
#         del generation
#         except Exception as e:
#             print(e)
#             if debug:
#                 import ipdb; ipdb.set_trace()
#             print('Returning last generation object')
#             return generation


    print(f'FINISHED. Returning best organism with name: {best_organism.name}')
    print(f'Finish time = {datetime.now().ctime()}\nStart time = {start_time}')
    return best_organism, generation