#ifndef EACS_H
#define EACS_H

#include "acs_parameters.h"
#include "tsp_instance.h"
#include "sop_instance.h"
#include "progress_visitor.h"
#include <iostream>
#include "ant_sop.h"
#include "problem_model.h"
#include "base_acs.h"

#include <unordered_map>


/*
 * An implementation of the Enhanced Ant Colony System
 */
template<typename NodeSelectionImpl, typename PheromoneMemory>
class EACS : public BaseACS {
public:
    using Problem_type = typename NodeSelectionImpl::Problem_type;
    using Parameters = ACSParameters;


    EACS(Parameters param,
        std::shared_ptr<NodeSelectionImpl> node_sel_impl,
        std::shared_ptr<PheromoneMemory> ph_mem,
        Problem_type& problem,
        RNG& rng)
        : BaseACS(rng),
        param_(param),
        node_sel_impl_(node_sel_impl),
        pheromone_memory_(ph_mem),
        problem_(problem),
        global_best_ant_(problem.get_dimension())
    {
    }


    void init() {
        node_sel_impl_->reset();
        pheromone_memory_->reset();
        stop_condition_->init();
    }


    void run_begin(std::shared_ptr<StopCondition> stop_condition) override {
        stop_condition_ = stop_condition;
        init();
        global_best_ant_.reset();
        global_best_found_iter_ = 0;
        gb_value_before_ls_ = global_best_ant_.get_value();
        seen_solutions_.clear();

        if (progress_visitor_) {
            progress_visitor_->started(*this);
        }
    }


    void run_next_iteration() override {
        build_ants_solutions();
        select_global_best_ant();
        pheromone_memory_->global_update(global_best_ant_.get_visited(),
            global_best_ant_.get_value(),
            param_.rho_);

        if (progress_visitor_) {
            progress_visitor_->iteration_done(*this);
        }
        stop_condition_->update(param_.ants_count_);
    }


    void run() override {
        while (!stop_condition_->is_reached()) {
            run_next_iteration();
        }
        if (progress_visitor_) {
            progress_visitor_->finished(*this);
        }
    }


    const Ant& get_global_best_ant() const { return global_best_ant_; }

    /*
     * Returns the number of iteration during which the current global best
     * solution was found.
     */
    uint64_t get_global_best_found_iter() const { return global_best_found_iter_; }


    uint64_t get_current_iter() const {
        if (!stop_condition_) {
            throw std::runtime_error("EACS::get_current_iter: No stop condition given");
        }
        return stop_condition_->get_iteration();
    }


    void set_progress_visitor(std::shared_ptr<ProgressVisitor<EACS<NodeSelectionImpl, PheromoneMemory>>> visitor) {
        progress_visitor_ = visitor;
    }


    uint32_t get_ants_count() const { return param_.ants_count_; }

private:

    void build_ants_solutions() {
        const auto dimension = problem_.get_dimension();

        ants_.clear();
        for (auto i = 0u; i < param_.ants_count_; ++i) {
            ants_.push_back(std::unique_ptr<Ant>{ create_ant(problem_) });
        }
        // Place each ant on a random start node
        for (auto& ant : ants_) {
            ant->move_to(rng_.rand_uint(0u, dimension - 1));
        }

        for (auto i = 0u; i < dimension - 1; ++i) {
            for (auto& ant : ants_) {
                perform_ant_move(ant.get());
                local_update(ant.get(), i, param_.phi_);
            }
        }
        // Local evaporation for last (closing) edge of each tour
        for (auto& ant : ants_) {
            assert(ant->has_complete_solution());
            const auto ant_value = ant->get_value();
            if (ant_value < gb_value_before_ls_) {
                gb_value_before_ls_ = ant_value;
            }
            local_update(ant.get(), dimension - 1, param_.phi_);
        }
        apply_local_search();
        eval_ants_solutions();
    }


    bool local_update(const Ant* ant, uint32_t node_index, double evap_ratio) {
        return pheromone_memory_->local_update(ant->get_visited(),
            node_index, evap_ratio);
    }


    void eval_ants_solutions() {
        for (auto& ant : ants_) {
            ant->set_value(problem_.eval_solution(ant->get_visited()));
        }
    }


    void select_global_best_ant() {
        // Now select best ant
        using namespace std;
        auto best_ant = get_best_ant(ants_);
        if (global_best_ant_.get_value() > best_ant->get_value()) {
            if (!problem_.is_solution_valid(best_ant->get_visited())) {
                std::cerr << "Invalid solution!" << std::endl;
                exit(EXIT_FAILURE);
            }
            global_best_ant_ = *best_ant;
            global_best_found_iter_ = stop_condition_->get_iteration();
            if (param_.verbose_) {
                std::cout << "Global best ant: " << global_best_ant_.get_value() << std::endl;
//                std::cout << "Seen solutions: " << seen_solutions_.size() << std::endl;
            }
        }
    }


    /*
     *  Move ant to a next node selected using pseudorandom proportional rule.
     *  Current implementation uses also candidates lists as proposed by M. Dorigo
     */
    void perform_ant_move(Ant* ant) {
        const auto sentinel = problem_.get_dimension();
        auto next_node = sentinel;

        const auto r = rng_.random();
        if (r > param_.q0_) {
            next_node = node_sel_impl_->select_next_node_p(*pheromone_memory_,
                ant,
                rng_);
        }
        else {
            // In the EACS the next node after the current one is the same as in the
            // global best solution unless it is already visited.
            if (has_global_best_ant()) {
                next_node = select_next_node_from_solution(ant, global_best_ant_.get_visited());
            }
            if (next_node == sentinel) { // The above may have failed
                next_node = node_sel_impl_->select_node_greedy(*pheromone_memory_, ant);
            }
        }
        assert(next_node != sentinel);
        ant->move_to(next_node);
    }


    /**
     * Applies LS but only to the ants which are not too far from the current
     * global best - this is a difference relative to the ACS.
     */
    void apply_local_search() {
        eval_ants_solutions(); // We have to do this now because we will need
                               // the solutions' values
        if (local_search_) {
            auto count = ants_.size();

            if (has_global_best_ant()) {
                // Apply LS only to solutions not too far from the current
                // global best
                //const auto best_value = gb_value_before_ls_;
                const auto best_value = global_best_ant_.get_value();
                const auto threshold = (1 + 0.2) * best_value; // max 20% from the current best
                auto i = 0u;
                for (auto j = i; j < ants_.size(); ++j) {
                    const auto ant_value = ants_[j]->get_value();
                    if (ant_value < threshold) {
                        std::swap(ants_[i], ants_[j]);
                        ++i;
                    }
                }
                count = i;
            }
            local_search_->apply(ants_, count, &global_best_ant_);
        }
    }


    bool has_global_best_ant() const {
        return global_best_ant_.get_value() != std::numeric_limits<double>::max();
    }


    uint32_t select_next_node_from_solution(Ant* ant, const std::vector<uint32_t>& route) const {
        const auto it = find(std::begin(route), std::end(route), ant->get_position());
        assert(it != std::end(route));
        const auto jt = std::next(it);
        const auto cand = (jt == end(route)) ? route.front() : *jt;
        const auto sentinel = problem_.get_dimension();
        return ant->is_available(cand) ? cand : sentinel;
    }


    Parameters param_;
    std::shared_ptr<NodeSelectionImpl> node_sel_impl_;
    std::shared_ptr<PheromoneMemory> pheromone_memory_;
    Problem_type& problem_;
    Ant global_best_ant_;
    uint64_t global_best_found_iter_;
    double gb_value_before_ls_;
    std::vector<std::unique_ptr<Ant>> ants_; // Pointers to Ants of the current iteration
    std::shared_ptr<ProgressVisitor<EACS<NodeSelectionImpl, PheromoneMemory>>> progress_visitor_ = nullptr;
    std::shared_ptr<StopCondition> stop_condition_ = nullptr;
    std::unordered_map<int64_t, int64_t> seen_solutions_;
};

#endif /* ifndef ACS_H */
