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

#include <deal.II/hp/fe_collection.h>
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
bool
face_has_ghost_penalty(const NonMatching::MeshClassifier<dim>                  &mesh_classifier,
                       const typename Triangulation<dim>::active_cell_iterator &cell,
                       const unsigned int                                       face_index)
{
  if (cell->at_boundary(face_index))
    return false;

  const NonMatching::LocationToLevelSet cell_location = mesh_classifier.location_to_level_set(cell);

  const NonMatching::LocationToLevelSet neighbor_location =
    mesh_classifier.location_to_level_set(cell->neighbor(face_index));

  if (cell_location == NonMatching::LocationToLevelSet::intersected &&
      neighbor_location != NonMatching::LocationToLevelSet::outside)
    return true;

  if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
      cell_location != NonMatching::LocationToLevelSet::outside)
    return true;

  return false;
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

  // const ConstraintType constraint_type = ConstraintType::approximate;
  const ConstraintType constraint_type = ConstraintType::Nitsche;
  // Set the alpha value for the exterior domain.
  // We are considering a large value to approximate a homogeneous Dirichlet BC.
  const double alpha_exterior = (constraint_type == ConstraintType::approximate) ? 10 : 1e-10;

  // Set the number of element refinements for performing a quadrature along
  // intersected cells.
  const int max_refinements = 5;

  // Parameters for Nitsche method
  const double dirichlet_bc      = 0.0;
  const double ghost_parameter   = 0.0; // default: 0.5
  const double nitsche_parameter = 5 * (fe_degree + 1) * fe_degree;


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

  // ghost penalty
  const QGauss<dim - 1>  face_quadrature(fe_degree + 1);
  FEInterfaceValues<dim> fe_interface_values(
    fe, face_quadrature, update_gradients | update_JxW_values | update_normal_vectors);

  // Nitsche terms
  const QGauss<1> quadrature_1D(fe_degree + 1);

  NonMatching::RegionUpdateFlags region_update_flags;
  region_update_flags.inside =
    update_values | update_gradients | update_JxW_values | update_quadrature_points;
  region_update_flags.surface = update_values | update_gradients | update_JxW_values |
                                update_quadrature_points | update_normal_vectors;

  hp::FECollection<dim> fe_collection(fe);

  // TODO: teach NonMatching::FEValues to take also a normal FiniteElement (?)
  // instead of a FECollection
  NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                    quadrature_1D,
                                                    region_update_flags,
                                                    mesh_classifier,
                                                    dof_handler,
                                                    signed_distance);

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

          // copy from step-85
          if (constraint_type == ConstraintType::Nitsche)
            {
              non_matching_fe_values.reinit(cell);

              const std::optional<NonMatching::FEImmersedSurfaceValues<dim>> &surface_fe_values =
                non_matching_fe_values.get_surface_fe_values();

              const double cell_side_length = cell->minimum_vertex_distance();

              if (surface_fe_values)
                {
                  for (const unsigned int q : surface_fe_values->quadrature_point_indices())
                    {
                      const Tensor<1, dim> &normal = surface_fe_values->normal_vector(q);
                      for (const unsigned int i : surface_fe_values->dof_indices())
                        {
                          for (const unsigned int j : surface_fe_values->dof_indices())
                            {
                              cell_matrix(i, j) += (-normal * surface_fe_values->shape_grad(i, q) *
                                                      surface_fe_values->shape_value(j, q) +
                                                    -normal * surface_fe_values->shape_grad(j, q) *
                                                      surface_fe_values->shape_value(i, q) +
                                                    nitsche_parameter / cell_side_length *
                                                      surface_fe_values->shape_value(i, q) *
                                                      surface_fe_values->shape_value(j, q)) *
                                                   surface_fe_values->JxW(q);
                            }

                          cell_rhs(i) += dirichlet_bc *
                                         (nitsche_parameter / cell_side_length *
                                            surface_fe_values->shape_value(i, q) -
                                          normal * surface_fe_values->shape_grad(i, q)) *
                                         surface_fe_values->JxW(q);
                        }
                    }
                }

              if (ghost_parameter > 0)
                {
                  // ghost penalty
                  for (const unsigned int f : cell->face_indices())
                    if (face_has_ghost_penalty(mesh_classifier, cell, f))
                      {
                        const unsigned int invalid_subface = numbers::invalid_unsigned_int;

                        fe_interface_values.reinit(cell,
                                                   f,
                                                   invalid_subface,
                                                   cell->neighbor(f),
                                                   cell->neighbor_of_neighbor(f),
                                                   invalid_subface);

                        const unsigned int n_interface_dofs =
                          fe_interface_values.n_current_interface_dofs();

                        FullMatrix<double> local_stabilization(n_interface_dofs, n_interface_dofs);

                        for (unsigned int q = 0; q < fe_interface_values.n_quadrature_points; ++q)
                          {
                            const Tensor<1, dim> normal = fe_interface_values.normal(q);
                            for (unsigned int i = 0; i < n_interface_dofs; ++i)
                              for (unsigned int j = 0; j < n_interface_dofs; ++j)
                                {
                                  local_stabilization(i, j) +=
                                    .5 * ghost_parameter * cell_side_length * normal *
                                    fe_interface_values.jump_in_shape_gradients(i, q) * normal *
                                    fe_interface_values.jump_in_shape_gradients(j, q) *
                                    fe_interface_values.JxW(q);
                                }
                          }

                        const std::vector<types::global_dof_index> local_interface_dof_indices =
                          fe_interface_values.get_interface_dof_indices();

                        std::cout << "local_dof_indices " << std::endl;
                        for (const auto &l : local_interface_dof_indices)
                          std::cout << l << " ";

                        std::cout << std::endl;
                        local_stabilization.print(std::cout);

                        constraints.distribute_local_to_global(local_stabilization,
                                                               local_interface_dof_indices,
                                                               system_matrix);
                      }
                }
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
