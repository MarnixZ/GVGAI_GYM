package tracks.singlePlayer.advanced.student;
import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;
import tracks.singlePlayer.tools.Heuristics.StateHeuristic;
import tracks.singlePlayer.tools.Heuristics.WinScoreHeuristic;

import java.util.*;

public class Agent extends AbstractPlayer {

    // variable
    private int POPULATION_SIZE = 10;
    private int SIMULATION_DEPTH = 5;
    private double DISCOUNT = 0.99; //0.99;
    private int CROSSOVER_TYPE = UNIFORM_CROSS;

    // set
    private boolean REEVALUATE = false;
    //    private boolean REPLACE = false;
    private int MUTATION = 1;
    private int TOURNAMENT_SIZE = 2;
    private int NO_PARENTS = 2;
    private int RESAMPLE = 1;
    private int ELITISM = 1;
    private int ROLLOUT_N = 5;

    // constants
    private final long BREAK_MS = 10;
    public static final double epsilon = 1e-6;
    static final int POINT1_CROSS = 0;
    static final int UNIFORM_CROSS = 1;

    private Individual[] population, nextPop;
    private int NUM_INDIVIDUALS;
    private int N_ACTIONS;
    private HashMap<Integer, Types.ACTIONS> action_mapping;

    private ElapsedCpuTimer timer;
    private Random randomGenerator;

    private StateHeuristic heuristic;
    private double acumTimeTakenEval = 0,avgTimeTakenEval = 0, avgTimeTaken = 0, acumTimeTaken = 0;
    private int numEvals = 0, numIters = 0;
    private boolean keepIterating = true;
    private long remaining;


    private Individual[] prev_pop = null;
    private HashMap<String, Types.ACTIONS> prev_obvs = new HashMap<String, Types.ACTIONS>();
    private StringBuilder sb = new StringBuilder();
    private boolean check_lookup = true;

    public int num_actions;
    public Types.ACTIONS[] actions;


    /**
     * Public constructor with state observation and time due.
     *
     * @param stateObs     state observation of the current game.
     * @param elapsedTimer Timer for the controller creation.
     */
    public Agent(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
        //Get the actions in a static array.
        ArrayList<Types.ACTIONS> act = stateObs.getAvailableActions();
        actions = new Types.ACTIONS[act.size()];
        for(int i = 0; i < actions.length; ++i)
        {
            actions[i] = act.get(i);
        }
        num_actions = actions.length;


        randomGenerator = new Random();
        heuristic = new WinScoreHeuristic(stateObs);
        this.timer = elapsedTimer;
    }

    @Override
    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
        this.timer = elapsedTimer;
        avgTimeTaken = 0;
        acumTimeTaken = 0;
        numEvals = 0;
        acumTimeTakenEval = 0;
        numIters = 0;
        remaining = timer.remainingTimeMillis();
        NUM_INDIVIDUALS = 0;
        keepIterating = true;

        // SEARCH IN LOOKUP TABLE FOR CURRENT OBSERVATION. IF FOUND, RETURN BEST MOVE.
        ArrayList<Observation> curr_obs[] = stateObs.getNPCPositions(stateObs.getAvatarPosition());
        sb.setLength(0);
        String sb_str = "";
        check_lookup = true;

        if (curr_obs != null) {
            for (Observation ooo : curr_obs[0]) {
                sb.append(ooo.toString());
            }
            sb.append(stateObs.getAvatarSpeed());
            sb.append(stateObs.getAvatarOrientation().toString());
            sb_str = sb.toString();
            if (prev_obvs.containsKey(sb_str)) {
                return prev_obvs.get(sb_str);
            }
        } else {
            check_lookup = false;
        }

        // INITIALISE POPULATION
        init_pop(stateObs);

        // RUN EVOLUTION
        remaining = timer.remainingTimeMillis();
        while (remaining > avgTimeTaken && remaining > BREAK_MS && keepIterating) {
            runIteration(stateObs);
            remaining = timer.remainingTimeMillis();
        }

        // SAVE PREVIOUS POPULATION
        prev_pop = new Individual[POPULATION_SIZE];
        for (int i = 0; i < population.length; i++) {
            prev_pop[i] = population[i];
        }

        // RETURN ACTION
        Types.ACTIONS best = get_best_action(population);


        // SAVE BEST MOVE IN LOOKUP TABLE
        if (check_lookup && !prev_obvs.containsKey(sb_str)) {
            prev_obvs.put(sb_str, best);
        }

        return best;
    }

    /**
     * Run evolutionary process for one generation
     * @param stateObs - current game state
     */
    private void runIteration(StateObservation stateObs) {
        ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();

        if (REEVALUATE) {
            for (int i = 0; i < ELITISM; i++) {
                if (remaining > 2*avgTimeTakenEval && remaining > BREAK_MS) { // if enough time to evaluate one more individual
                    evaluate(population[i], heuristic, stateObs);
                } else {keepIterating = false;}
            }
        }

        if (NUM_INDIVIDUALS > 1) {
            for (int i = ELITISM; i < NUM_INDIVIDUALS; i++) {
                if (remaining > 2*avgTimeTakenEval && remaining > BREAK_MS) { // if enough time to evaluate one more individual
                    Individual newind;

                    newind = crossover();
                    newind = newind.mutate(MUTATION);

                    // evaluate new individual, insert into population
                    add_individual(newind, nextPop, i, stateObs);

                    remaining = timer.remainingTimeMillis();

                } else {keepIterating = false; break;}
            }
            Arrays.sort(nextPop, new Comparator<Individual>() {
                @Override
                public int compare(Individual o1, Individual o2) {
                    if (o1 == null && o2 == null) {
                        return 0;
                    }
                    if (o1 == null) {
                        return 1;
                    }
                    if (o2 == null) {
                        return -1;
                    }
                    return o1.compareTo(o2);
                }
            });
        } else if (NUM_INDIVIDUALS == 1){
            Individual newind = new Individual(SIMULATION_DEPTH, N_ACTIONS, randomGenerator).mutate(MUTATION);
            evaluate(newind, heuristic, stateObs);
            if (newind.value > population[0].value)
                nextPop[0] = newind;
        }

        population = nextPop.clone();

        numIters++;
        acumTimeTaken += (elapsedTimerIteration.elapsedMillis());
        avgTimeTaken = acumTimeTaken / numIters;
    }

    /**
     * Evaluates an individual by rolling the current state with the actions in the individual
     * and returning the value of the resulting state; random action chosen for the opponent
     * @param individual - individual to be valued
     * @param heuristic - heuristic to be used for state evaluation
     * @param state - current state, root of rollouts
     * @return - value of last state reached
     */
    private double evaluate(Individual individual, StateHeuristic heuristic, StateObservation state) {

        ElapsedCpuTimer elapsedTimerIterationEval = new ElapsedCpuTimer();

        StateObservation st = state.copy();
        int i;
        double acum = 0, avg;
        for (i = 0; i < SIMULATION_DEPTH; i++) {
            if (! st.isGameOver()) {
                ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
                st.advance(action_mapping.get(individual.actions[i]));
                acum += elapsedTimerIteration.elapsedMillis();
                avg = acum / (i+1);
                remaining = timer.remainingTimeMillis();
                if (remaining < 2*avg || remaining < BREAK_MS) break;
            } else {
                break;
            }
        }

        StateObservation first = st.copy();
        double value = heuristic.evaluateState(first);

        // PERFORM MCTS ROLLOUT
        int k;
        double acum2 = 0, avg2;
        Random rd = new Random();
        for (k = 0; k < ROLLOUT_N; k++) {
            if (!st.isGameOver()) {
                int rd_state = state.getAvailableActions().size() + 1;
                ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();

                st.advance(action_mapping.get(rd.nextInt(rd_state)));

                acum2 += elapsedTimerIteration.elapsedMillis();
                avg2 = acum2 / (k + 1);
                remaining = timer.remainingTimeMillis();
                if (remaining < 2 * avg2 || remaining < BREAK_MS) break;
            } else {
                break;
            }
        }
        StateObservation second = st.copy();
        double value_2 = heuristic.evaluateState(second);



        // Apply discount factor
        value *= Math.pow(DISCOUNT,i);

        // UPDATE VALUE WITH MCTS ROLLOUT SCORE
        boolean gameOver = st.isGameOver();
        Types.WINNER win = st.getGameWinner();
        if (gameOver && win == Types.WINNER.PLAYER_WINS) {
            value *= Math.pow(1.15,((ROLLOUT_N-k)/2.0));
        } else if (gameOver && win == Types.WINNER.PLAYER_LOSES){
            value *= Math.pow(0.85,((ROLLOUT_N-k)/2.0));
        }

        individual.value = value;

        numEvals++;
        acumTimeTakenEval += (elapsedTimerIterationEval.elapsedMillis());
        avgTimeTakenEval = acumTimeTakenEval / numEvals;
        remaining = timer.remainingTimeMillis();

        return value;
    }


    /**
     * @return - the individual resulting from crossover applied to the specified population
     */
    private Individual crossover() {
        Individual newind = null;
        if (NUM_INDIVIDUALS > 1) {
            newind = new Individual(SIMULATION_DEPTH, N_ACTIONS, randomGenerator);
            Individual[] tournament = new Individual[TOURNAMENT_SIZE];
            Individual[] parents = new Individual[NO_PARENTS];

            ArrayList<Individual> list = new ArrayList<>();
            if (NUM_INDIVIDUALS > TOURNAMENT_SIZE) {
                list.addAll(Arrays.asList(population).subList(ELITISM, NUM_INDIVIDUALS));
            } else {
                list.addAll(Arrays.asList(population));
            }

            //Select a number of random distinct individuals for tournament and sort them based on value
            for (int i = 0; i < TOURNAMENT_SIZE; i++) {
                int index = randomGenerator.nextInt(list.size());
                tournament[i] = list.get(index);
                list.remove(index);
            }
            Arrays.sort(tournament);

            //get best individuals in tournament as parents
            if (NO_PARENTS <= TOURNAMENT_SIZE) {
                for (int i = 0; i < NO_PARENTS; i++) {
                    parents[i] = list.get(i);
                }
                newind.crossover(parents, CROSSOVER_TYPE);
            } else {
                System.out.println("WARNING: Number of parents must be LESS than tournament size.");
            }
        }
        return newind;
    }

    /**
     * Insert a new individual into the population at the specified position by replacing the old one.
     * @param newind - individual to be inserted into population
     * @param pop - population
     * @param idx - position where individual should be inserted
     * @param stateObs - current game state
     */
    private void add_individual(Individual newind, Individual[] pop, int idx, StateObservation stateObs) {
        evaluate(newind, heuristic, stateObs);
        pop[idx] = newind.copy();
    }

    /**
     * Initialize population
     * @param stateObs - current game state
     */
    private void init_pop(StateObservation stateObs) {

        double remaining = timer.remainingTimeMillis();

        N_ACTIONS = stateObs.getAvailableActions().size() + 1;
        action_mapping = new HashMap<>();
        int k = 0;
        for (Types.ACTIONS action : stateObs.getAvailableActions()) {
            action_mapping.put(k, action);
            k++;
        }
        action_mapping.put(k, Types.ACTIONS.ACTION_NIL);

        population = new Individual[POPULATION_SIZE];
        nextPop = new Individual[POPULATION_SIZE];

        if (prev_pop == null) {
            // WHEN GAME STARTS; POPULATION FOR FIRST MOVE
            for (int i = 0; i < POPULATION_SIZE; i++) {
                if (i == 0 || remaining > avgTimeTakenEval && remaining > BREAK_MS) {
                    population[i] = new Individual(SIMULATION_DEPTH, N_ACTIONS, randomGenerator);
                    evaluate(population[i], heuristic, stateObs);
                    remaining = timer.remainingTimeMillis();
                    NUM_INDIVIDUALS = i + 1;
                } else {
                    break;
                }
            }
        } else {
            // WHEN prev_pop IS KNOWN
            for (int i = 0; i < POPULATION_SIZE; i++) {
                if (i == 0 || remaining > avgTimeTakenEval && remaining > BREAK_MS) {
                    int[] moves = new int[SIMULATION_DEPTH];
                    for (int j = 0; j < SIMULATION_DEPTH; j++) {
                        if (j < SIMULATION_DEPTH-1 && prev_pop[i] != null) {
                            moves[j] = prev_pop[i].actions[j + 1];
                        } else {
                            moves[j] = randomGenerator.nextInt(N_ACTIONS);
                        }
                    }
                    population[i] = new Individual(SIMULATION_DEPTH, N_ACTIONS, randomGenerator);
                    population[i].setActions(moves);
                    evaluate(population[i], heuristic, stateObs);
                    remaining = timer.remainingTimeMillis();
                    NUM_INDIVIDUALS = i + 1;
                } else {
                    break;
                }
            }
        }


        if (NUM_INDIVIDUALS > 1)
            Arrays.sort(population, new Comparator<Individual>() {
                @Override
                public int compare(Individual o1, Individual o2) {
                    if (o1 == null && o2 == null) {
                        return 0;
                    }
                    if (o1 == null) {
                        return 1;
                    }
                    if (o2 == null) {
                        return -1;
                    }
                    return o1.compareTo(o2);
                }});
        for (int i = 0; i < NUM_INDIVIDUALS; i++) {
            if (population[i] != null)
                nextPop[i] = population[i].copy();
        }

    }

    /**
     * @param pop - last population obtained after evolution
     * @return - first action of best individual in the population (found at index 0)
     */
    private Types.ACTIONS get_best_action(Individual[] pop) {
        int bestAction = pop[0].actions[0];
        return action_mapping.get(bestAction);
    }

}
