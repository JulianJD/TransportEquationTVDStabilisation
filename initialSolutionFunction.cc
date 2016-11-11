
  using namespace dealii;

  //! initial values for the solution
  template <int dim>
  class InitialValueSolution : public Function<dim>
  {
  public:
    InitialValueSolution () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

  private:
	double radius(double x, double y, double x_0, double y_0) const;
  };

  // solution values
  template <int dim>
  double InitialValueSolution<dim>::value (const Point<dim>  &p,
                                     const unsigned int component) const
  {
    Assert (component == 0, ExcInternalError());
    Assert (dim == 2, ExcNotImplemented());

	const double PI  =3.141592653589793238463;
	double x = p[0];
	double y = p[1];
    // initial condition for the solid body rotation
	if ( (  radius( x, y, 0.5, 0.75 ) <= 1.0 ) &&
         ( ( std::abs( x - 0.5 ) >= 0.025 ) || ( y >= 0.85 ) ) ) {
		return 1.0;
	}
    else {
        if(  radius( x, y, 0.5, 0.25 ) <= 1.0 ) {
            return 1 - radius( x, y, 0.5, 0.25);
		}
        else {
            if ( radius( x, y, 0.25, 0.5) <= 1.0 ) {
                return 0.25*( 1 + std::cos( PI*radius( x, y, 0.25, 0.5)));
			}
            else {
                return 0.0;
            }
        }
    }
  }

  // auxiliary subroutine for the solid body rotation
  template <int dim>
  double InitialValueSolution<dim>::radius( double x, double y,
										    double x_0, double y_0) const {
  	return 1.0/0.15*std::sqrt( std::pow((x-x_0),2) + std::pow((y-y_0),2) );
  }
