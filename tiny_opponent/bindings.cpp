// pybind11 bindings for TinyOpponent.
//
// The object is created ONCE on the Python side and reused for every move.
// There is no per-move process start anywhere in this project.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tiny_opponent, m) {
    m.doc() = "TinyOpponent - tiny alpha-beta chess opponent for TonkAI";

    py::class_<tiny::Engine>(m, "TinyOpponent")
        .def(py::init([](int depth, int max_nodes, double randomness,
                         uint64_t seed, bool quiescence) {
                 tiny::Config cfg;
                 cfg.depth = depth;
                 cfg.max_nodes = max_nodes;
                 cfg.randomness = randomness;
                 cfg.seed = seed;
                 cfg.quiescence = quiescence;
                 return new tiny::Engine(cfg);
             }),
             py::arg("depth") = 2, py::arg("max_nodes") = 1000,
             py::arg("randomness") = 0.35, py::arg("seed") = 42,
             py::arg("quiescence") = true)
        .def("choose_move", &tiny::Engine::chooseMoveUci, py::arg("fen"),
             "Return a UCI move for the given FEN, or '0000' if none exists.")
        .def("legal_moves", &tiny::Engine::legalMovesUci, py::arg("fen"),
             "Engine-side legal moves (used by the tests to cross-check).")
        .def("set_seed", &tiny::Engine::setSeed, py::arg("seed"))
        .def("reset_stats", &tiny::Engine::resetStats)
        .def_property("depth", &tiny::Engine::depth, &tiny::Engine::setDepth)
        .def_property("max_nodes", &tiny::Engine::maxNodes,
                      &tiny::Engine::setMaxNodes)
        .def_property("randomness", &tiny::Engine::randomness,
                      &tiny::Engine::setRandomness)
        .def_property("quiescence", &tiny::Engine::quiescence,
                      &tiny::Engine::setQuiescence)
        .def_property_readonly(
            "nodes", [](const tiny::Engine& e) { return e.stats().nodes; })
        .def_property_readonly(
            "searches", [](const tiny::Engine& e) { return e.stats().searches; })
        .def_property_readonly("last_score",
                               [](const tiny::Engine& e) {
                                   return e.stats().last_score;
                               })
        .def_property_readonly("last_root_moves",
                               [](const tiny::Engine& e) {
                                   return e.stats().last_root_moves;
                               })
        .def_property_readonly("last_depth",
                               [](const tiny::Engine& e) {
                                   return e.stats().last_depth;
                               })
        .def_property_readonly("last_hit_node_cap",
                               [](const tiny::Engine& e) {
                                   return e.stats().last_hit_node_cap;
                               })
        .def("__repr__", [](const tiny::Engine& e) {
            return "<TinyOpponent depth=" + std::to_string(e.depth()) +
                   " max_nodes=" + std::to_string(e.maxNodes()) +
                   " randomness=" + std::to_string(e.randomness()) + ">";
        });

    m.def(
        "perft",
        [](const std::string& fen, int depth) {
            tiny::Position p;
            if (!p.setFen(fen)) throw std::invalid_argument("bad FEN: " + fen);
            return tiny::perft(p, depth);
        },
        py::arg("fen"), py::arg("depth"),
        "Count legal move sequences - the standard move-generation test.");

    m.def(
        "roundtrip_fen",
        [](const std::string& fen) {
            tiny::Position p;
            if (!p.setFen(fen)) throw std::invalid_argument("bad FEN: " + fen);
            return p.fen();
        },
        py::arg("fen"), "Parse and re-emit a FEN (used by the tests).");
}
