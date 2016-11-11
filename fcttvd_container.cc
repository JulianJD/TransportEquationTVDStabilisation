/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2016
 *
 * This file contains all necessary values for the FCT-TVD stabilisation
 *
 * ---------------------------------------------------------------------
 */

  using namespace dealii;

#include <vector>

  class FCTTVD_container
  {
  public:
    FCTTVD_container () {}
    void set_values ( const SparsityPattern  & sparsity_pattern,
                const SparseMatrix<double>* p_mass_matrix,
                const SparseMatrix<double>* p_lumped_mass_matrix,
                SparseMatrix<double>* p_convection_matrix );

    void art_diff_tvd(); // artificial diffusion tvd
    void lim_tvd( Vector<double> & solution, // antidiffusive flux-limiting
                  Vector<double> & rhs,
                  double time_step );

    void printContainerAttributes();

  private:
	SparseMatrix<double> d_matrix;
	const SparseMatrix<double>* p_mass_matrix;
	const SparseMatrix<double>* p_lumped_mass_matrix;

	SparseMatrix<double>* p_convection_matrix;
	std::vector< std::vector<int> > kedge;
	std::vector<double> dedge;
	int nedge;
  };


  // set_values
  void FCTTVD_container::set_values  ( const SparsityPattern  & sparsity_pattern,
                                       const SparseMatrix<double>* p_mass_matrix,
                                       const SparseMatrix<double>* p_lumped_mass_matrix,
                                       SparseMatrix<double>* p_convection_matrix )
  {

    d_matrix.reinit(sparsity_pattern);
    d_matrix = 0;
    this->p_mass_matrix = p_mass_matrix;
    this->p_lumped_mass_matrix = p_lumped_mass_matrix;
    this->p_convection_matrix = p_convection_matrix;

    int vec_size = p_mass_matrix->n_nonzero_elements();
    // kedge is 2xvec_size vector
	kedge.resize(2);
    for(int i = 0 ; i < 2 ; ++i)
    {
    	//Grow Columns by n
    	kedge[i].resize( vec_size );
    }
    // dedge is vec_size vector
	dedge.resize( vec_size );
	nedge = 0;
  }


  // artificial diffusion of the TVD scheme
  void FCTTVD_container::art_diff_tvd()
  {
    //typename
    SparseMatrix<double>::const_iterator mi = p_convection_matrix->begin(0);

    int row, col;
    double d_ij, dk_ij, dk_ji;
    int iedge;


    iedge = 0;
    // go through rows
    for (int m=0; m < p_convection_matrix->m(); m++ )
    {
        // go through columns
        // skip the diagonal entry
        // investigate only left-low part of the matrix
        for( mi = p_convection_matrix->begin(m)+1; mi != p_convection_matrix->end(m);  mi++)
        {
            if ( mi->column() > m )
            {
                break;
            }

            // set d_ij
            row = mi->row(); col = mi->column();
            d_ij = 0;
            dk_ij = p_convection_matrix->el(row, col);
            dk_ji = p_convection_matrix->el(col,row);
            d_ij = std::max( d_ij, (-dk_ij) );
            d_ij = std::max( d_ij, (-dk_ji) );

            // modification of the matrix entries
            p_convection_matrix->set(row, col, dk_ij + d_ij);
            p_convection_matrix->set(col, row, dk_ji + d_ij);
            p_convection_matrix->set(row, row, p_convection_matrix->diag_element(row) - d_ij);
            p_convection_matrix->set(col, col, p_convection_matrix->diag_element(col) - d_ij);

            // store necessary edges
            if( dk_ji > dk_ij )
            {
                kedge[0][iedge] = row;
                kedge[1][iedge] = col;
                dedge[iedge] = std::min( d_ij, dk_ji+d_ij );
            }
            else
            {
                kedge[0][iedge] = col;
                kedge[1][iedge] = row;
                dedge[iedge] = std::min( d_ij, dk_ij+d_ij );
            }
            iedge++;
        }

    } // end of th ematrix loop

    nedge = iedge;
  }


  // antidiffusive flux-limiting of the tvd scheme
  void FCTTVD_container::lim_tvd( Vector<double> & solution,
                                  Vector<double> & defect,
                                  double time_step )
  {
        //typename
        SparseMatrix<double>::const_iterator mi = p_convection_matrix->begin(0);

        int nvt = p_convection_matrix->m();

        // auxiliary variables
        std::vector<double> rp(nvt,0), rm(nvt,0);
        int iedge, i, j;
        double daux;
        double pp, pm, qp, qm;
        const double eps_loc = 0.0000000000000001;

        // Constructing fluxes
        for (iedge=0; iedge < nedge; iedge++)
        {
            i = kedge[0][iedge];
            j = kedge[1][iedge];

            daux = dedge[iedge]*( solution[i] - solution[j] );

            rp[i] += std::max( 0.0, daux);
            rm[i] += std::min( 0.0, daux);
            //std::cout << " iedge = " << iedge << " nedge=" << nedge << std::endl;
        }

        //std::cout << "col=" <<  nvt << "    ...done" << std::endl;
        for( i=0; i<nvt; i++)
        {
            pp=rp[i];
            pm=rm[i];

            qp=0.0;
            qm=0.0;

            rp[i]=0.0;
            rm[i]=0.0;

            // loop here
            for( mi = p_convection_matrix->begin(i)+1; mi != p_convection_matrix->end(i);  mi++)
            {
                j = mi->column();
                daux = p_convection_matrix->el(i,j)*( solution[j] - solution[i] );

                qp += std::max( 0.0, daux);
                qm += std::min( 0.0, daux);
            }

            if( pp > eps_loc )
            {
                rp[i] = std::min( 1.0, qp/pp );
            }
            if( pm < -eps_loc )
            {
                rm[i] = std::min( 1.0, qm/pm );
            }
        }

        // correction of the low-order solution
        for (iedge=0; iedge < nedge; iedge++)
        {
            // node numbers for the current edge
            i = kedge[0][iedge];
            j = kedge[1][iedge];

            // antidiffusive flux to be limited
            daux = dedge[iedge]*( solution[i] - solution[j]);

            if( daux > 0 )
            {
                daux = time_step*rp[i]*daux;
            }
            else
            {
                daux = time_step*rm[i]*daux;
            }

            // high-order resolution
            defect[i] += daux;
            defect[j] -= daux;
        }

        // vector deallocation
        rp.clear();
        //
        rm.clear();
  }


  // set_values
  void FCTTVD_container::printContainerAttributes  ( )
  {
      std::cout << std::endl << " Printinting the container attributes ... " << std::endl;

      // print nedge
      std::cout << " nedge=" <<  nedge << std::endl;

      std::cout << std::endl << "Dedge values: " << std::endl;
      for ( int i=0; i< nedge; i++)
      {
          std::cout << " i=" << i << " dedge=" << dedge[i] << std::endl;
      }

      std::cout << std::endl << "Kedge values: " << std::endl;
      for ( int i=0; i< nedge; i++)
      {
          std::cout << " i=" << i << " kedge_1=" << kedge[0][i] << " kedge_2=" << kedge[1][i] << std::endl;
      }

  }
