/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */


// The program starts with the usual include files, all of which you should
// have seen before by now:
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/base/function_lib.h>
#include <deal.II/lac/block_vector.h>

#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

// for the transport term: from the step-12 (advective equation)
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

#include <fstream>
#include <iostream>

// include the convection/advection parts
#include <deal.II/base/tensor_function.h>
#include <deal.II/numerics/error_estimator.h>

// contains tvd stabilisation code
#include "fcttvd_container.cc"
// include the initial condition for the solution
#include "initialSolutionFunction.cc"


/**
* the problem name space
**/
namespace HeatEq
{
  using namespace dealii;

  // problem parameters
  const unsigned int dimension = 2;

  // AS: mesh information
  const unsigned int initial_global_refinement = 0; //2
  const unsigned int n_adaptive_pre_refinement_steps = 7; //4

  // the main class of this application
  template<int dim>
  class HeatEquation
  {
  public:
    HeatEquation();
    void run();

  private:
    void setup_system();
    void solve_time_step();
    void output_results() const;
    void tensor1_output() const;


    void lump_mass_matrix ();
    void assemble_convection ();

    //! problem relevant parameters
    Triangulation<dim>   triangulation;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;
    ConstraintMatrix     constraints;
    SparsityPattern      sparsity_pattern;

    void refine_mesh (const unsigned int min_grid_level,
                      const unsigned int max_grid_level);


    //! time-related parameters
    double               time;
    double               time_step;
    unsigned int         timestep_number;

    //! equation-related parameters
    const double         theta;
    double               D_Laplace; // magnitude of the Laplacian: $D\Delta u$
    double               K_Convection; // magnitude of convection:
									   // $K_Convection v \cdot \nabla u$

    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> lumped_mass_matrix;

    SparseMatrix<double> laplace_matrix;

	SparseMatrix<double> convection_matrix;
    // fct/tvd stabilisation-related class
    friend class FCTTVD_container;

	// some auxiliary stuctures and functions
    struct AssemblyScratchData
    {
      AssemblyScratchData (const FiniteElement<dim> &fe);
      AssemblyScratchData (const AssemblyScratchData &scratch_data);

      FEValues<dim>     fe_values;
    };
    struct AssemblyCopyData
    {
      FullMatrix<double> cell_matrix;
      std::vector<types::global_dof_index> local_dof_indices;
    };
    void local_assemble_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
                        AssemblyScratchData &scratch,
                        AssemblyCopyData    &copy_data);
    void copy_local_to_global (const AssemblyCopyData &copy_data);

    // stabilisation stuff for the convection
	FCTTVD_container fcttvd_container;

    SparseMatrix<double> system_matrix;
    Vector<double>       solution;
    Vector<double>       old_solution;
    Vector<double>       system_rhs;
    Vector<double>       system_defect;

  };


   /**
   * advection tensor (v_1, ..., v_dim)^T
   **/
  template <int dim>
  class AdvectionField : public TensorFunction<1,dim>
  {
  public:
    AdvectionField () : TensorFunction<1,dim> () {}

    virtual Tensor<1,dim> value (const Point<dim> &p) const;

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<1,dim> >    &values) const;

    // corresponding exception
    DeclException2 (ExcDimensionMismatch,
                    unsigned int, unsigned int,
                    << "The vector has size " << arg1 << " but should have "
                    << arg2 << " elements.");
  };


  // value of advection field at a point
  template <int dim>
  Tensor<1,dim>
  AdvectionField<dim>::value (const Point<dim> &p) const
  {
    Point<dim> value;
    //value[0] = p[0];
    value[0] = 0.5 - p[1];
    for (unsigned int i=1; i<dim; ++i)
    {
        //value[1] = p[1];
        value[1] = p[0] - 0.5;
    }

    return value;
  }


  // value_list
  template <int dim>
  void
  AdvectionField<dim>::value_list (const std::vector<Point<dim> > &points,
                                   std::vector<Tensor<1,dim> >    &values) const
  {
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));

    for (unsigned int i=0; i<points.size(); ++i)
      values[i] = AdvectionField<dim>::value (points[i]);
  }


  // assembly scratch data
  template <int dim>
  HeatEquation<dim>::AssemblyScratchData::
  AssemblyScratchData (const FiniteElement<dim> &fe)
    :
    fe_values (fe,
               QSimpson<dim>(),
               update_values   | update_gradients |
               update_quadrature_points | update_JxW_values) //,
  {}


  // assembly scratch data 2
  template <int dim>
  HeatEquation<dim>::AssemblyScratchData::
  AssemblyScratchData (const AssemblyScratchData &scratch_data)
    :
    fe_values (scratch_data.fe_values.get_fe(),
               scratch_data.fe_values.get_quadrature(),
               update_values   | update_gradients |
               update_quadrature_points | update_JxW_values) //,
  {}


  // local assembly of a matrix
  template <int dim>
  void
  HeatEquation<dim>::
  local_assemble_system (const typename DoFHandler<dim>::active_cell_iterator &cell,
                         AssemblyScratchData                                  &scratch_data,
                         AssemblyCopyData                                     &copy_data)
  {
    // declaration
    const AdvectionField<dim> advection_field;

    // Then we define some abbreviations to avoid unnecessarily long lines:
    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_q_points      = scratch_data.fe_values.get_quadrature().size();


    copy_data.cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
    copy_data.local_dof_indices.resize(dofs_per_cell);

    std::vector<Tensor<1,dim> > advection_directions (n_q_points);

    scratch_data.fe_values.reinit (cell);

    advection_field.value_list (scratch_data.fe_values.get_quadrature_points(),
                                advection_directions);

    // loop other quadrature points
    for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              // assembly the local matrix entries
              copy_data.cell_matrix(i,j) += advection_directions[q_point]*
                                            scratch_data.fe_values.shape_grad(i,q_point)*
                                            scratch_data.fe_values.shape_value(j,q_point)*
                                            scratch_data.fe_values.JxW(q_point);
            }

    // local to global
    cell->get_dof_indices (copy_data.local_dof_indices);
  }


  // local to global
  template <int dim>
  void
  HeatEquation<dim>::copy_local_to_global (const AssemblyCopyData &copy_data)
  {
    for (unsigned int i=0; i<copy_data.local_dof_indices.size(); ++i)
      {
        for (unsigned int j=0; j<copy_data.local_dof_indices.size(); ++j)
          convection_matrix.add (copy_data.local_dof_indices[i],
                                 copy_data.local_dof_indices[j],
                                 copy_data.cell_matrix(i,j));
      }
  }


  /**
   * forcing term -> Right Hand Side
   **/
  template<int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide ()
      :
      Function<dim>()
      //,
      //period (0.2)
    {}

    virtual double value (const Point<dim> &p,
                          const unsigned int component = 0) const;
  };


  // values
  template<int dim>
  double RightHandSide<dim>::value (const Point<dim> &p,
                                    const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());
    Assert (dim == 2, ExcNotImplemented());

	return 0;
  }


   /**
   * boundary function
   **/
  template<int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value (const Point<dim>  &p,
                          const unsigned int component = 0) const;
  };



  template<int dim>
  double BoundaryValues<dim>::value (const Point<dim> &/*p*/,
                                     const unsigned int component) const
  {
    Assert(component == 0, ExcInternalError());
    return 0;
  }


   /**
   * main constructor
   **/
  template<int dim>
  HeatEquation<dim>::HeatEquation ()
    :
    fe(1),
    dof_handler(triangulation),
    time_step(0.001),
	theta(1.0),
	D_Laplace(0.0),
	K_Convection(1.0)
  {}


   /**
   * set-up of the system
   **/
  template<int dim>
  void HeatEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "==========================================="
              << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    constraints.clear ();


    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);


    mass_matrix.reinit(sparsity_pattern);

    lumped_mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);

	// convection matrix and necessary stabilisation manipulations
	convection_matrix.reinit(sparsity_pattern);
	fcttvd_container.set_values ( sparsity_pattern,
                                  &mass_matrix,
                                  &lumped_mass_matrix,
                                  &convection_matrix );

    system_matrix.reinit(sparsity_pattern);

	//! set up the phase-field values
	const int meshLevel = initial_global_refinement +
   						  n_adaptive_pre_refinement_steps;

	//! mass_matrix and etc.
    MatrixCreator::create_mass_matrix(dof_handler,
                                      //QGauss<dim>(fe.degree+1),
                                      QSimpson<dim>(),
                                      mass_matrix); //,
 									  //&phaseFieldB);
    // create lumped_mass_matrix using the mass_matrix
    lump_mass_matrix ();
	// laplacian
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         //QGauss<dim>(fe.degree+1),
                                         QSimpson<dim>(),
                                         laplace_matrix); //,
										 //&phaseFieldB);

    //! initiliasation of solution and etc.
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    system_defect.reinit(dof_handler.n_dofs());
  }


  /**
   * construction of the lumped mass matrix
   **/
  template <int dim>
  void HeatEquation<dim>::lump_mass_matrix ()
  {
    lumped_mass_matrix = 0;

    //typename
   SparseMatrix<double>::const_iterator mi = mass_matrix.begin(0);

    double row_sum;
    // go through rows
    for (int m=0; m < mass_matrix.m(); m++ )
    {

        row_sum = 0;
        // go through columns
        for ( mi = mass_matrix.begin(m); mi != mass_matrix.end(m);  mi++)
        {
            row_sum += mi->value();
        }

        lumped_mass_matrix.set( m,m, row_sum);
    }
  }


  /**
   * assembly of the convection term method
   **/
  template <int dim>
  void HeatEquation<dim>::assemble_convection ()
  {
     AssemblyScratchData scratch_data(fe);
     AssemblyCopyData copy_data;
     typename DoFHandler<dim>::active_cell_iterator cell_it = dof_handler.begin_active(),
                                                    endc = dof_handler.end();
     for (; cell_it!=endc; ++cell_it)
      {
        local_assemble_system ( cell_it, scratch_data, copy_data);

        copy_local_to_global (copy_data);
      }

  }


  /**
   * solve the time-step method
   **/
  template<int dim>
  void HeatEquation<dim>::solve_time_step()
  {
    //SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
    //SolverControl solver_control(1000, 1e-8 * system_defect.l2_norm());
    SolverControl solver_control (1000, 1e-12);

    //SolverCG<> cg(solver_control);
	SolverGMRES<> gmres(solver_control);
    //SolverBicgstab<> bicgstab(solver_control);

    PreconditionSSOR<> preconditioner; // gmres
    //PreconditionJacobi<> preconditioner; // preconditioner for the SolverBicgstab
    preconditioner.initialize(system_matrix, 1.0);

    //! gmres
    gmres.solve(system_matrix, solution, system_rhs, preconditioner);

    //! bicgstab
    //bicgstab.solve( system_matrix, solution, system_defect, preconditioner);
    //solution.add( 1, old_solution);

    constraints.distribute(solution);

    std::cout << "     " << solver_control.last_step()
              // << " CG iterations." << std::endl;
         	  // AS: solver is changed into GMRES
              << " GMRES iterations." << std::endl;
  }


  /**
   * output-result method
   **/
  template<int dim>
  void HeatEquation<dim>::output_results() const
  {

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(solution, "U");

    data_out.build_patches();

    const std::string filename = "solution-"
                                 + Utilities::int_to_string(timestep_number, 3) +
                                 ".gmv";
    std::ofstream output(filename.c_str());
    data_out.write_gmv(output);
  }


  /**
   * tensor-output method
   **/
  template<int dim>
  void HeatEquation<dim>::tensor1_output() const
  {

    DataOut<dim> data_out;
    Vector<double> vec_output;

    DoFHandler<dim>      local_dof_handler(triangulation);
    FESystem<dim>        my_fe(FE_Q<dim>(1), dim);
    local_dof_handler.distribute_dofs (my_fe);

    vec_output.reinit ( local_dof_handler.n_dofs() );

    VectorTools::interpolate( local_dof_handler,
                               VectorFunctionFromTensorFunction<dim>( AdvectionField<dim>(), 0, dim),
                               vec_output );

    std::vector<std::string> solution_names(dim, "vec");
    for (int i=0; i<dim; i++)
    {
        solution_names[i] += Utilities::int_to_string(i+1, 1);
    }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim, DataComponentInterpretation::component_is_part_of_vector);

    data_out.attach_dof_handler( local_dof_handler );

    data_out.add_data_vector ( vec_output, solution_names,
                          DataOut<dim>::type_dof_data,
                          data_component_interpretation);

    data_out.build_patches();
    const std::string filename = "solution-"
                                 + Utilities::int_to_string(timestep_number, 3) +
                                 ".gmv";
    std::ofstream output(filename.c_str());
    data_out.write_gmv(output);

    // release
    solution_names.clear();
    data_out.clear();
    local_dof_handler.clear();

  }


  /**
   * refine-mesh
   **/
  template <int dim>
  void HeatEquation<dim>::refine_mesh (const unsigned int min_grid_level,
                                       const unsigned int max_grid_level)
  {
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        QGauss<dim-1>(fe.degree+1),
                                        typename FunctionMap<dim>::type(),
                                        solution,
                                        estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_fraction (triangulation,
                                                       estimated_error_per_cell,
                                                       0.6, 0.4);

    if (triangulation.n_levels() > max_grid_level)
      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active(max_grid_level);
           cell != triangulation.end(); ++cell)
        cell->clear_refine_flag ();
    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active(min_grid_level);
         cell != triangulation.end_active(min_grid_level); ++cell)
      cell->clear_coarsen_flag ();

    // solution transfer
    SolutionTransfer<dim> solution_trans(dof_handler);

    Vector<double> previous_solution;
    previous_solution = solution;
    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

    // coarsening and refinement
    triangulation.execute_coarsening_and_refinement ();
    setup_system ();

    solution_trans.interpolate(previous_solution, solution);
    constraints.distribute (solution);
  }


  /**
   * run method
   */
  template<int dim>
  void HeatEquation<dim>::run()
  {
    // GridGenerator::hyper_L (triangulation);
	GridGenerator::hyper_cube (triangulation);
    //triangulation.refine_global(initial_global_refinement);
	// AS: fine refinement without any tricky stuff
	triangulation.refine_global(initial_global_refinement +
								n_adaptive_pre_refinement_steps);

    setup_system();

    unsigned int pre_refinement_step = 0;

    Vector<double> tmp;
    Vector<double> forcing_terms;

start_time_iteration:

    tmp.reinit (solution.size());
    forcing_terms.reinit (solution.size());

	// set initial values for solution and old_solution
    VectorTools::interpolate(dof_handler,
                             // ZeroFunction<dim>(),
							 InitialValueSolution<dim>(),
                             old_solution);
    solution = old_solution;

    timestep_number = 0;
    time            = 0;

	//! output
     output_results();
    // tensor1_output();

    //! time-loop 6.283 0.001
    while (time < 6.283)
    {
        time += time_step;
        ++timestep_number;

        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

		//! assembly of the convection term
		convection_matrix = 0;
		assemble_convection ();
		//! adding the artificial diffusion
		fcttvd_container.art_diff_tvd();

        //! construction of the right-hand side
        // mass_matrix.vmult(system_rhs, old_solution);
        lumped_mass_matrix.vmult(system_rhs, old_solution);

        laplace_matrix.vmult(tmp, old_solution);
        system_rhs.add(-(1 - theta) * time_step * D_Laplace, tmp);

        //! assembly of the forcing term
        // \left[ \Delta t (1-\theta)f^{n-1} + \Delta t \theta f^n \right]$.
        RightHandSide<dim> rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler,
                                            //QGauss<dim>(fe.degree+1),
                                            QSimpson<dim>(),
                                            rhs_function,
                                            tmp);
        forcing_terms = tmp;
        forcing_terms *= time_step * theta;

        rhs_function.set_time(time - time_step);
        VectorTools::create_right_hand_side(dof_handler,
                                            //QGauss<dim>(fe.degree+1),
                                            QSimpson<dim>(),
                                            rhs_function,
                                            tmp);
        forcing_terms.add(time_step * (1 - theta), tmp);

        //! forcing term -> right-hand-side
        system_rhs += forcing_terms;

        //! assembly of the system-matrix
        // system_matrix.copy_from(mass_matrix);
        system_matrix.copy_from(lumped_mass_matrix);
        system_matrix.add(theta * time_step * D_Laplace, laplace_matrix);

		// convection term
        system_matrix.add( -theta * time_step * K_Convection, convection_matrix);

        //! construction the defect
//        system_matrix.vmult( system_defect, old_solution);
//        system_defect *= -1;
//        system_defect.add( 1, system_rhs);

		//! anti-diffusive flux-limiting
//		fcttvd_container.lim_tvd( solution,
//                                  system_defect,
//                                  time_step );
		fcttvd_container.lim_tvd( solution,
                                  system_rhs,
                                  time_step );

        constraints.condense (system_matrix, system_rhs);
        //constraints.condense (system_matrix, system_defect);

        //! solve the current-time linear system
        solve_time_step();

		//! output of the current time-instance
		if( timestep_number % 10 == 0 ) {
	        output_results();
	        // tensor1_output();
		}


        old_solution = solution;

      } // end of time-loop
  } // end of run
} // namespace HeatEq


//! main function
int main()
{
  try
    {
      using namespace dealii;
      using namespace HeatEq;

      HeatEquation<dimension> heat_equation_solver;
      heat_equation_solver.run();

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}




