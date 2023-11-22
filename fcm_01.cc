#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/quadrature_selector.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <mpi.h>

#include <fstream>
#include <iostream>

using namespace dealii;

enum ConstraintType
{
  approximate,
  Nitsche
};

namespace dealii::GridGenerator
{
  template <int dim, typename VectorType>
  void
  create_triangulation_with_marching_cube_algorithm(Triangulation<dim - 1, dim> &tria,
                                                    const Mapping<dim>          &mapping,
                                                    const DoFHandler<dim> &background_dof_handler,
                                                    const VectorType      &ls_vector,
                                                    const double           iso_level,
                                                    const unsigned int     n_subdivisions = 1,
                                                    const double           tolerance      = 1e-10)
  {
    std::vector<Point<dim>>        vertices;
    std::vector<CellData<dim - 1>> cells;
    SubCellData                    subcelldata;

    const GridTools::MarchingCubeAlgorithm<dim, VectorType> mc(mapping,
                                                               background_dof_handler.get_fe(),
                                                               n_subdivisions,
                                                               tolerance);

    const bool vector_is_ghosted = ls_vector.has_ghost_elements();

    if (vector_is_ghosted == false)
      ls_vector.update_ghost_values();

    mc.process(background_dof_handler, ls_vector, iso_level, vertices, cells);

    if (vector_is_ghosted == false)
      ls_vector.zero_out_ghost_values();

    std::vector<unsigned int> considered_vertices;

    // note: the following operation does not work for simplex meshes yet
    // GridTools::delete_duplicated_vertices (vertices, cells, subcelldata,
    // considered_vertices);

    if (vertices.size() > 0)
      tria.create_triangulation(vertices, cells, subcelldata);
  }
} // namespace dealii::GridGenerator

// This mini program shows a basic implementation of the finite cell method (FCM)
// as presented e.g. in
//
// Parvizian, Jamshid, Alexander Düster, and Ernst Rank. "Finite cell method:
// h-and p-extension for embedded domain problems in solid mechanics."
// Computational Mechanics 41.1 (2007): 121-133.
//
// As a demonstration example we consider the Poisson equation
//
//     -Δu = 1   on Ω,
//       u = 0   on dΩ.

template <int dim>
void
exact_solution()
{
  ConditionalOStream pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

  pcout << "Start FEM test for dim=" << dim << std::endl;
  TimerOutput timer(MPI_COMM_WORLD, pcout, TimerOutput::summary, TimerOutput::wall_times);
  // create mesh
  const int fe_degree         = 1;
  const int global_refinement = 6;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_ball(tria, Point<dim>(), 0.5);
  tria.refine_global(global_refinement);
  pcout << "    - number of finite elements: " << tria.n_global_active_cells() << std::endl;

  // create finite element system
  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler;
  dof_handler.reinit(tria);
  dof_handler.distribute_dofs(fe);

  // setup constraints
  timer.enter_subsection("setup_constraints_sparsity");
  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
  constraints.close();

  // initialize sparse matrix
  TrilinosWrappers::SparseMatrix system_matrix;
  IndexSet                       locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  TrilinosWrappers::SparsityPattern dsp;
  dsp.reinit(dof_handler.locally_owned_dofs(),
             dof_handler.locally_owned_dofs(),
             locally_relevant_dofs,
             MPI_COMM_WORLD);

  DoFTools::make_sparsity_pattern(
    dof_handler, dsp, constraints, true, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
  dsp.compress();
  system_matrix.reinit(dsp);
  timer.leave_subsection();

  // assemble weak form of vector depending on state
  LinearAlgebra::distributed::Vector<double> rhs, solution;
  rhs.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);
  solution.reinit(rhs);

  timer.enter_subsection("assemble");
  FEValues<dim>                        fe_values(fe,
                          QGauss<dim>(fe_degree + 1),
                          update_values | update_gradients | update_quadrature_points |
                            update_JxW_values);
  const unsigned int                   n_dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double>                       cell_rhs(n_dofs_per_cell);
  FullMatrix<double>                   cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_rhs    = 0;
          cell_matrix = 0;

          fe_values.reinit(cell);

          for (const unsigned int q_index : fe_values.quadrature_point_indices())
            for (const unsigned int i : fe_values.dof_indices())
              {
                cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                fe_values.JxW(q_index));            // dx
                                                                    //

                for (const unsigned int j : fe_values.dof_indices())
                  cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                                        fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                                        fe_values.JxW(q_index));
              }

          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices, system_matrix, rhs);
        }
    }

  rhs.compress(VectorOperation::add);
  system_matrix.compress(VectorOperation::add);


  timer.leave_subsection();

  // solve system
  timer.enter_subsection("solve_linear_system");
  ReductionControl                                     solver_control(1000, 1e-10, 1e-10);
  SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);
  solution.reinit(rhs);

  // setup preconditioner
  TrilinosWrappers::PreconditionJacobi preconditioner;
  preconditioner.initialize(system_matrix);

  solver.solve(system_matrix, solution, rhs, preconditioner);
  timer.leave_subsection();

  pcout << "    - solved in " << solver_control.last_step() << " iterations." << std::endl;


  if (true)
    {
      MappingQ1<dim>        mapping;
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = (dim > 1);

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      data_out.add_data_vector(dof_handler, rhs, "rhs_exact");
      data_out.add_data_vector(dof_handler, solution, "solution_exact");
      data_out.build_patches();
      std::string output = "out_exact.vtu";
      data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
    }
}


template <int dim>
void
test()
{
  ConditionalOStream pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
  pcout << "Start FCM test for dim=" << dim << std::endl;

  TimerOutput timer(MPI_COMM_WORLD, pcout, TimerOutput::summary, TimerOutput::wall_times);
  // create mesh
  const int fe_degree         = 1;
  const int global_refinement = 4;

  const ConstraintType constraint_type = ConstraintType::approximate;
  // Set the alpha value for the exterior domain.
  // We are considering a large value to approximate a homogeneous Dirichlet BC.
  const double alpha_exterior = 10;

  // Set the number of element refinements for performing a quadrature along
  // intersected cells.
  const int max_refinements = 5;

  parallel::shared::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(global_refinement);

  MappingQ1<dim> mapping;

  // create finite element system
  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler;
  dof_handler.reinit(tria);
  dof_handler.distribute_dofs(fe);

  // assemble weak form of vector depending on state
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  LinearAlgebra::distributed::Vector<double> rhs;
  rhs.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);

  // create level-set field and mesh classifier
  LinearAlgebra::distributed::Vector<double> signed_distance;
  signed_distance.reinit(rhs);

  timer.enter_subsection("mesh_classifier");
  const Functions::SignedDistance::Sphere<dim> signed_distance_sphere(Point<dim>(), 0.5);
  VectorTools::interpolate(dof_handler, signed_distance_sphere, signed_distance);
  signed_distance.update_ghost_values();

  NonMatching::MeshClassifier<dim> mesh_classifier(dof_handler, signed_distance);
  mesh_classifier.reclassify();
  timer.leave_subsection();

  timer.enter_subsection("setup_constraints_sparsity");

  // In addition to choose a large value for alpha for material points being
  // external of the domain, we explicitly enforce homogeneous Dirichlet
  // boundary conditions along (outside) element faces of intersected cells.
  AffineConstraints<double> constraints;

  if (constraint_type == ConstraintType::approximate)
    {
      // prepare for Dirichlet BC --> collect DoF indices
      FEValues<dim> fe_values(fe,
                              Quadrature<dim>(fe.get_unit_support_points()),
                              update_values | update_quadrature_points | update_JxW_values);

      std::vector<double>                  phi_at_q(fe_values.get_quadrature().size());
      const unsigned int                   n_dofs_per_cell = fe.n_dofs_per_cell();
      std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

      std::vector<types::global_dof_index> bc_indices;

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            {
              const auto cell_location = mesh_classifier.location_to_level_set(cell);

              if (cell_location == NonMatching::LocationToLevelSet::intersected)
                {
                  fe_values.reinit(cell);

                  // state of quadrature_point
                  fe_values.get_function_values(signed_distance, phi_at_q);
                  cell->get_dof_indices(local_dof_indices);

                  for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                    if (phi_at_q[i] >= 0)
                      bc_indices.emplace_back(local_dof_indices[i]);
                }
            }
        }

      // add entries to constraint matrix
      for (const auto &bc : bc_indices)
        if (!constraints.is_constrained(bc))
          constraints.add_constraint(bc, {}, 0);

      constraints.close();
    }
  else if (constraint_type == ConstraintType::Nitsche)
    {
    }

  // add contributions from nitsche
  // data structures for marching-cube algorithm
  const QGauss<dim == 1 ? 1 : dim - 1> surface_quad(dof_handler.get_fe().degree + 1);

  GridTools::MarchingCubeAlgorithm<dim, LinearAlgebra::distributed::Vector<double>> mc(
    MappingQ1<dim>(), fe);


  // initialize sparse matrix
  TrilinosWrappers::SparseMatrix    system_matrix;
  TrilinosWrappers::SparsityPattern dsp;
  {
    dsp.reinit(dof_handler.locally_owned_dofs(),
               dof_handler.locally_owned_dofs(),
               locally_relevant_dofs,
               MPI_COMM_WORLD);

    DoFTools::make_sparsity_pattern(
      dof_handler, dsp, constraints, true, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    dsp.compress();
    system_matrix.reinit(dsp);
  }
  timer.leave_subsection();

  timer.enter_subsection("assemble");
  // finite element for intersected cells
  FEValues<dim> fe_values_intersected(fe,
                                      QIterated<dim>(QGauss<1>(fe_degree + 1), max_refinements),
                                      update_values | update_gradients | update_quadrature_points |
                                        update_JxW_values);

  // finite element for other cells
  FEValues<dim> fe_values(fe,
                          QGauss<dim>(fe_degree + 1),
                          update_values | update_gradients | update_quadrature_points |
                            update_JxW_values);

  unsigned int count_intersect = 0;
  unsigned int count_inside    = 0;

  FEValues<dim> *fe_values_used = nullptr;

  const unsigned int                   n_dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double>                       cell_rhs(n_dofs_per_cell);
  FullMatrix<double>                   cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

  std::vector<double> buffer(n_dofs_per_cell);

  std::vector<Point<dim>> quadrature_points;


  FEPointEvaluation<1, dim> fe_values_immersed(mapping, fe, update_values | update_gradients);


  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          const auto cell_location = mesh_classifier.location_to_level_set(cell);

          if (cell_location == NonMatching::LocationToLevelSet::intersected)
            {
              fe_values_used = &fe_values_intersected;
              count_intersect += 1;
            }
          else
            {
              fe_values_used = &fe_values;
            }

          if (cell_location == NonMatching::LocationToLevelSet::inside)
            count_inside += 1;

          cell_rhs    = 0;
          cell_matrix = 0;

          fe_values_used->reinit(cell);

          // state of quadrature_point
          std::vector<double> phi_at_q(fe_values_used->get_quadrature().size());
          fe_values_used->get_function_values(signed_distance, phi_at_q);
          for (auto &phi : phi_at_q)
            phi = (phi <= 0) ? 1.0 : alpha_exterior;

          for (const unsigned int q_index : fe_values_used->quadrature_point_indices())
            {
              quadrature_points.emplace_back(fe_values_used->quadrature_point(q_index));
              for (const unsigned int i : fe_values_used->dof_indices())
                {
                  cell_rhs(i) += (fe_values_used->shape_value(i, q_index) * // phi_i(x_q)
                                  phi_at_q[q_index] *                       // alpha
                                  fe_values_used->JxW(q_index));            // dx

                  for (const unsigned int j : fe_values_used->dof_indices())
                    cell_matrix(i, j) +=
                      (fe_values_used->shape_grad(i, q_index) * // grad phi_i(x_q)
                       fe_values_used->shape_grad(j, q_index) * // grad phi_j(x_q)
                       phi_at_q[q_index] *                      // alpha
                       fe_values_used->JxW(q_index));
                }
            }

          if (constraint_type == ConstraintType::Nitsche &&
              cell_location == NonMatching::LocationToLevelSet::intersected)
            {
              // Determine the quadrature
              // point location (at the reference cell) and weight
              // determine points and cells of aux surface triangulation
              std::vector<Point<dim>>                       surface_vertices;
              std::vector<CellData<dim == 1 ? 1 : dim - 1>> surface_cells;

              // run marching cube algorithm
              if (dim > 1)
                mc.process_cell(
                  cell, signed_distance, 0 /*contour_value*/, surface_vertices, surface_cells);
              else
                mc.process_cell(cell, signed_distance, 0 /*contour_value*/, surface_vertices);

              if (surface_vertices.size() == 0)
                continue; // cell is not cut by interface -> no quadrature points have the be
                          // determined

              std::vector<Point<dim>> points_real;
              std::vector<Point<dim>> points;
              std::vector<double>     weights;

              // create aux triangulation of subcells
              Triangulation<dim == 1 ? 1 : dim - 1, dim> surface_triangulation;
              surface_triangulation.create_triangulation(surface_vertices, surface_cells, {});

              FE_Nothing<dim == 1 ? 1 : dim - 1, dim> fe_nothing;
              FEValues<dim == 1 ? 1 : dim - 1, dim>   fe_eval(
                fe_nothing, surface_quad, update_quadrature_points | update_JxW_values);

              // loop over all cells ...
              for (const auto &sub_cell : surface_triangulation.active_cell_iterators())
                {
                  fe_eval.reinit(sub_cell);

                  // ... and collect quadrature points and weights
                  for (const auto &q : fe_eval.quadrature_point_indices())
                    {
                      points_real.emplace_back(fe_eval.quadrature_point(q));
                      points.emplace_back(
                        mapping.transform_real_to_unit_cell(cell, fe_eval.quadrature_point(q)));
                      weights.emplace_back(fe_eval.JxW(q));
                    }
                }
              const unsigned int n_points = points.size();

              const ArrayView<const Point<dim>> unit_points(points.data(), n_points);
              const ArrayView<const double>     JxW(weights.data(), n_points);

              fe_values_immersed.reinit(cell, unit_points);

              for (unsigned int q = 0; q < n_points; ++q)
                {
                  // Use the gradient of the level set field to compute the normal vector
                  constraints.get_dof_values(signed_distance,
                                             local_dof_indices.begin(),
                                             buffer.begin(),
                                             buffer.end());
                  fe_values_immersed.evaluate(buffer,
                                              EvaluationFlags::values | EvaluationFlags::gradients);
                  const auto unit_normal =
                    fe_values_immersed.get_gradient(q) /
                    std::max(fe_values_immersed.get_gradient(q).norm(), 1e-6);

                  // get values of the shape functions
                  std::fill(buffer.begin(), buffer.end(), 1);
                  fe_values_immersed.evaluate(buffer, EvaluationFlags::values);

                  const auto result_gradient =
                    -unit_normal * fe_values_immersed.get_value(q) * JxW[q];

                  // TODO: Multiplication by test function and assembly
                  // fe_values_immersed.submit_gradient(result_gradient, q);
                }

              // integrate
              // fe_values_immersed.test_and_sum(buffer, EvaluationFlags::values);
              // cell_rhs += buffer;
            }


          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices, system_matrix, rhs);
        }
    }
  rhs.compress(VectorOperation::add);
  system_matrix.compress(VectorOperation::add);
  timer.leave_subsection();
  pcout << "    - number of finite cells: " << tria.n_global_active_cells() << std::endl;
  pcout << "    - number of inside cells: " << Utilities::MPI::sum(count_inside, MPI_COMM_WORLD)
        << std::endl;
  pcout << "    - number of intersected cells: "
        << Utilities::MPI::sum(count_intersect, MPI_COMM_WORLD) << std::endl;

  // solve system
  timer.enter_subsection("solve_linear_system");
  ReductionControl                                     solver_control(1000, 1e-10, 1e-10);
  SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);
  LinearAlgebra::distributed::Vector<double>           solution;
  solution.reinit(signed_distance);

  // setup preconditioner
  TrilinosWrappers::PreconditionJacobi preconditioner;
  preconditioner.initialize(system_matrix);

  solver.solve(system_matrix, solution, rhs, preconditioner);
  timer.leave_subsection();
  pcout << "    - solved in " << solver_control.last_step() << " iterations." << std::endl;

  if (true)
    {
      MappingQ1<dim>        mapping;
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = (dim > 1);

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      data_out.add_data_vector(dof_handler, signed_distance, "signed_distance");
      data_out.add_data_vector(dof_handler, rhs, "rhs_cm");
      data_out.add_data_vector(dof_handler, solution, "solution_fcm");
      data_out.build_patches();
      std::string output = "out.vtu";
      data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

      // write quadrature points to file
      quadrature_points = Utilities::MPI::reduce<std::vector<Point<dim>>>(
        quadrature_points, MPI_COMM_WORLD, [](const auto &a, const auto &b) {
          auto result = a;
          result.insert(result.end(), b.begin(), b.end());
          return result;
        });
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::ofstream file("quadrature_points.csv");
          for (const auto &p : quadrature_points)
            file << p << std::endl;
          file.close();
        }
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  test<2>();
  exact_solution<2>();
  return 0;
}
