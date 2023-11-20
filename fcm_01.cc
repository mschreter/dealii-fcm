#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_selector.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <mpi.h>

#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim>
void
test()
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "start test for dim=" << dim << std::endl;

  // create mesh
  const int fe_degree = 2;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, -1, 1);
  tria.refine_global(4);

  // create finite element system
  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler;
  dof_handler.reinit(tria);
  dof_handler.distribute_dofs(fe);

  // setup constraints
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


  // element refinements inside the domain
  const int max_refinements = 2;

  // create level-set field and mesh classifier
  LinearAlgebra::distributed::Vector<double> signed_distance;
  signed_distance.reinit(dof_handler.locally_owned_dofs(),
                         DoFTools::extract_locally_active_dofs(dof_handler),
                         MPI_COMM_WORLD);

  const Functions::SignedDistance::Sphere<dim> signed_distance_sphere(Point<dim>(), 0.5);
  VectorTools::interpolate(dof_handler, signed_distance_sphere, signed_distance);
  NonMatching::MeshClassifier<dim> mesh_classifier(dof_handler, signed_distance);
  mesh_classifier.reclassify();

  // assemble weak form of vector depending on state
  LinearAlgebra::distributed::Vector<double> rhs;
  rhs.reinit(signed_distance);

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

  FEValues<dim> *fe_values_used = nullptr;

  const unsigned int                   n_dofs_per_cell = fe.n_dofs_per_cell();
  Vector<double>                       cell_rhs(n_dofs_per_cell);
  FullMatrix<double>                   cell_matrix(n_dofs_per_cell, n_dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

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

          cell_rhs    = 0;
          cell_matrix = 0;

          fe_values_used->reinit(cell);

          // state of quadrature_point
          std::vector<double> phi_at_q(fe_values_used->get_quadrature().size());
          fe_values_used->get_function_values(signed_distance, phi_at_q);
          for (auto &phi : phi_at_q)
            phi = (phi <= 0) ? 1.0 : 1e-10;

          for (const unsigned int q_index : fe_values_used->quadrature_point_indices())
            for (const unsigned int i : fe_values_used->dof_indices())
              {
                cell_rhs(i) += (fe_values_used->shape_value(i, q_index) * // phi_i(x_q)
                                phi_at_q[q_index] *                       // alpha
                                fe_values_used->JxW(q_index));            // dx
                                                                          //

                for (const unsigned int j : fe_values_used->dof_indices())
                  cell_matrix(i, j) += (fe_values_used->shape_grad(i, q_index) * // grad phi_i(x_q)
                                        fe_values_used->shape_grad(j, q_index) * // grad phi_j(x_q)
                                        phi_at_q[q_index] *                      // alpha
                                        fe_values_used->JxW(q_index));
              }

          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices, system_matrix, rhs);
        }
    }

  rhs.compress(VectorOperation::add);
  system_matrix.compress(VectorOperation::add);

  // solve system
  ReductionControl                                     solver_control(1000, 1e-10, 1e-10);
  SolverCG<LinearAlgebra::distributed::Vector<double>> solver(solver_control);
  LinearAlgebra::distributed::Vector<double>           solution;
  solution.reinit(signed_distance);

  // setup preconditioner
  TrilinosWrappers::PreconditionJacobi preconditioner;
  preconditioner.initialize(system_matrix);

  solver.solve(system_matrix, solution, rhs, preconditioner);

  if (true)
    {
      MappingQ1<dim>        mapping;
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = (dim > 1);

      DataOut<dim> data_out;
      data_out.set_flags(flags);

      data_out.add_data_vector(dof_handler, signed_distance, "signed_distance");
      data_out.add_data_vector(dof_handler, rhs, "rhs");
      data_out.add_data_vector(dof_handler, solution, "solution");
      data_out.build_patches();
      std::string output = "out.vtu";
      data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
    }

  std::cout << "number of intersected cells: "
            << Utilities::MPI::sum(count_intersect, MPI_COMM_WORLD) << std::endl;
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  test<2>();
  return 0;
}
