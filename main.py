
from genetic_algorithm.generation.generation import Generation



# CUDA_VISIBLE_DEVICES=i wandb agent xxxx

def main(data, config, best_organism = None, verbose=True, debug=False):
    print(config)
    for phase in range(config.generation.num_phases):
        print("PHASE {}".format(phase))
        try:
            generation = Generation(data=data,
                                    generation_config=config['generation'],
                                    organism_config=config['organism'],
                                    phase=phase,
                                    previous_best_organism=best_organism,
                                    verbose=verbose,
                                    debug=debug)

            best_organism = generation.run_phase()
        except Exception as e:
            print(e)
            print('Returning last generation object')
            return generation
        
    print(f'FINISHED. Returning best organism with name: {best_organism.name}')
    return best_organism