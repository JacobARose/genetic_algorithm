
from genetic_algorithm.generation.generation import Generation





def main(data, config, best_organism = None, verbose=True, debug=False):
    print(config)
    for phase in range(config.generation.num_phases):
        print("PHASE {}".format(phase))
        generation = Generation(data=data,
                                generation_config=config['generation'],
                                organism_config=config['organism'],
                                phase=phase,
                                previous_best_organism=best_organism,
                                verbose=verbose,
                                DEBUG=debug)

        best_organism = generation.run_phase()