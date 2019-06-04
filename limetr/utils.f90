! fortran helper functions for speed up

MODULE varmat

CONTAINS

! public functions
! -----------------------------------------------------------------------------
FUNCTION dot_mv(v, Z, d, study_sizes, x, m, n, k) RESULT(y)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: m, n, k
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k), x(n)
    INTEGER(8), INTENT(IN) :: study_sizes(m)
    REAL(8) :: y(n)
    INTEGER(8) :: a, b, i

    a = 1
    b = 0
    DO i = 1, m
        b = b + study_sizes(i)
        CALL block_dot_mv(v(a:b), Z(a:b, :), d, x(a:b), y(a:b), &
            study_sizes(i), k)
        a = a + study_sizes(i)
    END DO

END FUNCTION dot_mv


FUNCTION dot_mm(v, Z, d, study_sizes, X, m, n, k, l) RESULT(Y)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: m, n, k, l
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k), X(n, l)
    INTEGER(8), INTENT(IN) :: study_sizes(m)
    REAL(8) :: Y(n, l)
    INTEGER(8) :: a, b, i

    a = 1
    b = 0
    DO i = 1, m
        b = b + study_sizes(i)
        CALL block_dot_mm(v(a:b), Z(a:b, :), d, X(a:b, :), Y(a:b, :), &
            study_sizes(i), k, l)
        a = a + study_sizes(i)
    END DO

END FUNCTION dot_mm


FUNCTION invdot_mv(v, Z, d, study_sizes, x, m, n, k) RESULT(y)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: m, n, k
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k), x(n)
    INTEGER(8), INTENT(IN) :: study_sizes(m)
    REAL(8) :: y(n)
    INTEGER(8) :: a, b, i

    a = 1
    b = 0
    DO i = 1, m
        b = b + study_sizes(i)
        IF (n <= k) THEN
            CALL block_invdot_mv_n(v(a:b), Z(a:b, :), d, x(a:b), y(a:b), &
                study_sizes(i), k)
        ELSE
            CALL block_invdot_mv_k(v(a:b), Z(a:b, :), d, x(a:b), y(a:b), &
                study_sizes(i), k)
        END IF
        a = a + study_sizes(i)
    END DO

END FUNCTION invdot_mv


FUNCTION invdot_mm(v, Z, d, study_sizes, X, m, n, k, l) RESULT(Y)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: m, n, k, l
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k), X(n, l)
    INTEGER(8), INTENT(IN) :: study_sizes(m)
    REAL(8) :: Y(n, l)
    INTEGER(8) :: a, b, i

    a = 1
    b = 0
    DO i = 1, m
        b = b + study_sizes(i)
        IF (n <= k) THEN
            CALL block_invdot_mm_n(v(a:b), Z(a:b, :), d, X(a:b, :), Y(a:b, :),&
                study_sizes(i), k, l)
        ELSE
            CALL block_invdot_mm_k(v(a:b), Z(a:b, :), d, X(a:b, :), Y(a:b, :),&
                study_sizes(i), k, l)
        END IF
        a = a + study_sizes(i)
    END DO

END FUNCTION invdot_mm


FUNCTION logdet(v, Z, d, study_sizes, m, n, k) RESULT(val)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: m, n, k
    REAL(8), INTENT(IN) :: v(n), Z(n,k), d(k)
    INTEGER(8), INTENT(IN) :: study_sizes(m)
    REAL(8) :: val
    INTEGER(8) :: a, b, i

    val = 0.D0
    a = 1
    b = 0
    DO i = 1, m
        b = b + study_sizes(i)
        IF (n <= k) THEN
            val = val + block_logdet_n(v(a:b), Z(a:b, :), d, study_sizes(i), k)
        ELSE
            val = val + block_logdet_k(v(a:b), Z(a:b, :), d, study_sizes(i), k)
        END IF
        a = a + study_sizes(i)
    END DO

END FUNCTION logdet

! private SUBROUTINEs
! -----------------------------------------------------------------------------
SUBROUTINE block_dot_mv(v, Z, d, x, y, n, k)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: n, k
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k), x(n)
    REAL(8), INTENT(INOUT) :: y(n)

    y = v*x + MATMUL(Z, d*MATMUL(TRANSPOSE(Z), x))

END SUBROUTINE block_dot_mv


SUBROUTINE block_dot_mm(v, Z, d, X, Y, n, k, l)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: n, k, l
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k), X(n, l)
    REAL(8), INTENT(INOUT) :: Y(n, l)
    REAL(8) :: T(n, k)
    INTEGER(8) :: i, j

    DO j = 1, l
        Y(:, j) = X(:, j)*v
    END DO

    DO i = 1, n
        T(i, :) = Z(i, :)*d
    END DO

    Y = Y + MATMUL(T, MATMUL(TRANSPOSE(Z), X))

END SUBROUTINE block_dot_mm


SUBROUTINE block_invdot_mv_n(v, Z, d, x, y, n, k)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: n, k
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k), x(n)
    REAL(8), INTENT(INOUT) :: y(n)
    REAL(8) :: M(n, n), T(n, k)
    INTEGER(8) :: i, info, ipiv(n)

    ! compute covariance matrix
    DO i = 1, n
        T(i, :) = Z(i, :)*d
    END DO

    M = MATMUL(T, TRANSPOSE(Z))

    DO i = 1, n
        M(i, i) = M(i, i) + v(i)
    END DO

    ! solve linear system
    y = x
    CALL DGESV(n, 1, M, n, ipiv, y, n, info)

END SUBROUTINE block_invdot_mv_n


SUBROUTINE block_invdot_mv_k(v, Z, d, x, y, n, k)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: n, k
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k), x(n)
    REAL(8), INTENT(INOUT) :: y(n)
    REAL(8) :: M(k, k), T(n, k), u(k)
    INTEGER(8) :: j, info, ipiv(k)

    ! compute covariance matrix
    DO j = 1, k
        T(:, j) = Z(:, j)/v
    END DO

    M = MATMUL(TRANSPOSE(Z), T)

    DO j = 1, k
        M(j, j) = M(j, j) + 1.D0/d(j)
    END DO

    ! solve linear system
    u = MATMUL(TRANSPOSE(T), x)
    CALL DGESV(k, 1, M, k, ipiv, u, k, info)

    y = x/v - MATMUL(T, u)

END SUBROUTINE block_invdot_mv_k


SUBROUTINE block_invdot_mm_n(v, Z, d, X, Y, n, k, l)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: n, k, l
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k), X(n, l)
    REAL(8), INTENT(INOUT) :: Y(n, l)
    REAL(8) :: M(n, n), T(n, k)
    INTEGER(8) :: i, info, ipiv(n)

    ! compute covariance matrix
    DO i = 1, n
        T(i, :) = Z(i, :)*d
    END DO

    M = MATMUL(T, TRANSPOSE(Z))

    DO i = 1, n
        M(i, i) = M(i, i) + v(i)
    END DO

    ! solve linear system
    Y = X
    CALL DGESV(n, l, M, n, ipiv, Y, n, info)

END SUBROUTINE block_invdot_mm_n


SUBROUTINE block_invdot_mm_k(v, Z, d, X, Y, n, k, l)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: n, k, l
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k), X(n, l)
    REAL(8), INTENT(INOUT) :: Y(n, l)
    REAL(8) :: M(k, k), T(n, k), U(k, l)
    INTEGER(8) :: j, info, ipiv(k)

    ! compute covariance matrix
    DO j = 1, k
        T(:, j) = Z(:, j)/v
    END DO

    M = MATMUL(TRANSPOSE(Z), T)

    DO j = 1, k
        M(j, j) = M(j, j) + 1.D0/d(j)
    END DO

    ! solve linear system
    U = MATMUL(TRANSPOSE(T), X)
    CALL DGESV(k, l, M, k, ipiv, U, k, info)

    Y = MATMUL(T, U)

    DO j = 1, l
        Y(:, j) = X(:, j)/v - Y(:, j)
    END DO

END SUBROUTINE block_invdot_mm_k


FUNCTION block_logdet_n(v, Z, d, n, k) RESULT(val)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: n, k
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k)
    REAL(8) :: val
    REAL(8) :: M(n, n), T(n, k)
    INTEGER(8) :: i, info, ipiv(n)

    ! compute covariance matrix
    DO i = 1, n
        T(i, :) = Z(i, :)*d
    END DO

    M = MATMUL(T, TRANSPOSE(Z))

    DO i = 1, n
        M(i, i) = M(i, i) + v(i)
    END DO

    ! compute the determinant
    CALL DGETRF(n, n, M, n, ipiv, info)

    val = 0.D0
    DO i = 1, n
        val = val + LOG(ABS(M(i, i)))
    END DO

END FUNCTION block_logdet_n


FUNCTION block_logdet_k(v, Z, d, n, k) RESULT(val)
    IMPLICIT NONE
    INTEGER(8), INTENT(IN) :: n, k
    REAL(8), INTENT(IN) :: v(n), Z(n, k), d(k)
    REAL(8) :: val
    REAL(8) :: M(k, k), T(n, k)
    INTEGER(8) :: i, j, info, ipiv(k)

    ! compute covariance matrix
    DO i = 1, n
        T(i, :) = Z(i, :)*d
    END DO

    DO j = 1, k
        T(:, j) = T(:, j)/v
    END DO

    M = MATMUL(TRANSPOSE(Z), T)

    DO j = 1, k
        M(j, j) = M(j, j) + 1.D0
    END DO

    ! compute the determinant
    CALL DGETRF(k, k, M, k, ipiv, info)

    val = 0.D0
    DO i = 1, n
        val = val + LOG(v(i))
    END DO

    DO j = 1, k
        val = val + LOG(ABS(M(j, j)))
    END DO

END FUNCTION block_logdet_k


END MODULE varmat