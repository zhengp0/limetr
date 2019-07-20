! fortran helper functions for special mat calculation


MODULE IZMAT


CONTAINS


! LSVD compute the svd of a matrix and output the singular value and the left
! singular vector
! -----------------------------------------------------------------------------
SUBROUTINE LSVD(sz1, sz2, su, ss, z, u, s)
    IMPLICIT NONE
    INTEGER(8) :: sz1, sz2, su, ss
    REAL(8) :: z(sz1, sz2), u(su), s(ss)
    INTEGER(8) :: lwork, info
    REAL(8), allocatable :: work(:)

    !f2py INTENT(IN) :: z
    !f2py INTENT(HIDE), DEPEND(z) :: sz1 = SIZE(z, 1), sz2 = SIZE(z, 2)
    !f2py INTENT(HIDE), DEPEND(u) :: su = SIZE(u)
    !f2py INTENT(HIDE), DEPEND(s) :: sp = SIZE(s)
    !f2py INTENT(INOUT) :: u, s

    lwork = 2*MAX(1, 3*ss + MAX(sz1, sz2), 5*ss)

    ALLOCATE(work(lwork))
    CALL DGESVD('S', 'N', sz1, sz2, z, sz1, s, u, sz1, z, ss, &
                work, lwork, info)
    DEALLOCATE(work)

END SUBROUTINE LSVD


! ZDECOMP compute the svd of the block of Z matrix
! -----------------------------------------------------------------------------
SUBROUTINE ZDECOMP(sz1, sz2, su, ss, m, nz, nu, ns, z, u, s)
    IMPLICIT NONE
    INTEGER(8) :: sz1, sz2, su, ss, m
    INTEGER(8) :: nz(m), nu(m), ns(m)
    REAL(8) :: z(sz1, sz2), u(su), s(ss)
    INTEGER(8) :: az, bz, au, bu, as, bs, i

    !f2py INTENT(IN) :: nz, nu, ns, z
    !f2py INTENT(HIDE), DEPEND(nz) :: m = SIZE(nz)
    !f2py INTENT(HIDE), DEPEND(z) :: sz1 = SIZE(z, 1), sz2 = SIZE(z, 2)
    !f2py INTENT(HIDE), DEPEND(u) :: su = SIZE(u)
    !f2py INTENT(HIDE), DEPEND(s) :: ss = SIZE(s)
    !f2py INTENT(INOUT) :: u, s

    az = 1
    au = 1
    as = 1
    bz = 0
    bu = 0
    bs = 0

    DO i = 1, m
        bz = bz + nz(i)
        bu = bu + nu(i)
        bs = bs + ns(i)

        CALL LSVD(nz(i), sz2, nu(i), ns(i), z(az:bz, :), u(au:bu), s(as:bs))

        az = az + nz(i)
        au = au + nu(i)
        as = as + ns(i)
    END DO

END SUBROUTINE ZDECOMP

! BLOCK_IZMV calculate matrix vector multiplication of block izmat
! -----------------------------------------------------------------------------
SUBROUTINE BLOCK_IZMV(su, sd, sx, u, d, x, y)
    IMPLICIT NONE
    INTEGER(8) :: su, sd, sx
    REAL(8) :: u(su), d(sd), x(sx), y(sx)
    REAL(8) :: c(sd)

    !f2py INTENT(IN) :: u, d, x
    !f2py INTENT(HIDE), DEPEND(u) :: su = SIZE(u)
    !f2py INTENT(HIDE), DEPEND(d) :: sd = SIZE(d)
    !f2py INTENT(HIDE), DEPEND(x) :: sx = SIZE(x)
    !f2py INTENT(INOUT) :: y

    CALL DGEMV('T', sx, sd, 1.d0, u, sx, x, 1, 0.d0, c, 1)
    c = c * d
    CALL DGEMV('N', sx, sd, 1.d0, u, sx, c, 1, 0.d0, y, 1)
    y = y + x

END SUBROUTINE BLOCK_IZMV

! IZMV calculate matrix vector multiplication of izmat
! -----------------------------------------------------------------------------
FUNCTION IZMV(su, sd, sx, m, nu, nd, nx, u, d, x) RESULT(y)
    IMPLICIT NONE
    INTEGER(8) :: su, sd, sx, m
    INTEGER(8) :: nu(m), nd(m), nx(m)
    REAL(8) :: u(su), d(sd), x(sx)
    REAL(8) :: c(sd), y(sx)
    INTEGER(8) :: ax, bx, au, bu, ad, bd, i

    !f2py INTENT(IN) :: u, d, x
    !f2py INTENT(HIDE), DEPEND(u) :: su = SIZE(u)
    !f2py INTENT(HIDE), DEPEND(d) :: sd = SIZE(d)
    !f2py INTENT(HIDE), DEPEND(x) :: sx = SIZE(x)
    !f2py INTENT(HIDE), DEPEND(nx) :: m = SIZE(nx)

    ax = 1
    au = 1
    ad = 1
    bx = 0
    bu = 0
    bd = 0

    DO i = 1, m
        bx = bx + nx(i)
        bu = bu + nu(i)
        bd = bd + nd(i)

        CALL BLOCK_IZMV(nu(i), nd(i), nx(i), u(au:bu), d(ad:bd), x(ax:bx), &
                        y(ax:bx))

        ax = ax + nx(i)
        au = au + nu(i)
        ad = ad + nd(i)
    END DO

END FUNCTION IZMV


END MODULE IZMAT
