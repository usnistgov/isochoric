// To go in derivs:

if (mode == STEP_PARAMETRIC) {
                // Assume to start that we want the positive sign for dt / dp
                TYPE dtdp = sqrt((drhovec_dpL.array().pow(2) + drhovec_dpV.array().pow(2)).sum());
                // Calculate the vector of derivatives of molar concentration w.r.t. arclength
                EigenArray drhoLdt = drhovec_dpL / dtdp, drhoVdt = drhovec_dpV / dtdp;
                TYPE c0 = (get_forwards_integration()) ? 1 : -1;
                TYPE c = c0;
                // Are the derivatives opposite to what they used to be?
                bool dot_negative = (drhoLdt.matrix().dot(this->drhoLdt_old.matrix()) > 0);
                // Check if the sign needs to be flipped
                if (drhoLdt_old.size() == 0) {}
                else if (dot_negative) { c0 *= -1; }
                dtdp *= c;
               
                // Our independent variable is the arc length t
                Eigen::Map<Eigen::VectorXd> drhovec_dtL_wrap(&(f[0]), N),
                                            drhovec_dtV_wrap(&(f[0]) + N, N);
                drhovec_dtL_wrap = drhovec_dpL/dtdp;
                drhovec_dtV_wrap = drhovec_dpV/dtdp;

                // Temporarily store the derivatives, will be picked up in post_deriv_callback
                drhovec_dtL_store = drhovec_dtL_wrap;
                drhovec_dtV_store = drhovec_dtV_wrap;
            }
            else