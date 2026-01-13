*> \brief \b DSTEQR
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*> Download DSTEQR + dependencies
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dsteqr.f">
*> [TGZ]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dsteqr.f">
*> [ZIP]</a>
*> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dsteqr.f">
*> [TXT]</a>
*
*  Definition:
*  ===========
*
*       SUBROUTINE DSTEQR( COMPZ, N, D, E, Z, LDZ, WORK, INFO )
*
*       .. Scalar Arguments ..
*       CHARACTER          COMPZ
*       INTEGER            INFO, LDZ, N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION   D( * ), E( * ), WORK( * ), Z( LDZ, * )
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> DSTEQR computes all eigenvalues and, optionally, eigenvectors of a
*> symmetric tridiagonal matrix using the implicit QL or QR method.
*> The eigenvectors of a full or band symmetric matrix can also be found
*> if DSYTRD or DSPTRD or DSBTRD has been used to reduce this matrix to
*> tridiagonal form.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] COMPZ
*> \verbatim
*>          COMPZ is CHARACTER*1
*>          = 'N':  Compute eigenvalues only.
*>          = 'V':  Compute eigenvalues and eigenvectors of the original
*>                  symmetric matrix.  On entry, Z must contain the
*>                  orthogonal matrix used to reduce the original matrix
*>                  to tridiagonal form.
*>          = 'I':  Compute eigenvalues and eigenvectors of the
*>                  tridiagonal matrix.  Z is initialized to the identity
*>                  matrix.
*> \endverbatim
*>
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>          The order of the matrix.  N >= 0.
*> \endverbatim
*>
*> \param[in,out] D
*> \verbatim
*>          D is DOUBLE PRECISION array, dimension (N)
*>          On entry, the diagonal elements of the tridiagonal matrix.
*>          On exit, if INFO = 0, the eigenvalues in ascending order.
*> \endverbatim
*>
*> \param[in,out] E
*> \verbatim
*>          E is DOUBLE PRECISION array, dimension (N-1)
*>          On entry, the (n-1) subdiagonal elements of the tridiagonal
*>          matrix.
*>          On exit, E has been destroyed.
*> \endverbatim
*>
*> \param[in,out] Z
*> \verbatim
*>          Z is DOUBLE PRECISION array, dimension (LDZ, N)
*>          On entry, if  COMPZ = 'V', then Z contains the orthogonal
*>          matrix used in the reduction to tridiagonal form.
*>          On exit, if INFO = 0, then if  COMPZ = 'V', Z contains the
*>          orthonormal eigenvectors of the original symmetric matrix,
*>          and if COMPZ = 'I', Z contains the orthonormal eigenvectors
*>          of the symmetric tridiagonal matrix.
*>          If COMPZ = 'N', then Z is not referenced.
*> \endverbatim
*>
*> \param[in] LDZ
*> \verbatim
*>          LDZ is INTEGER
*>          The leading dimension of the array Z.  LDZ >= 1, and if
*>          eigenvectors are desired, then  LDZ >= max(1,N).
*> \endverbatim
*>
*> \param[out] WORK
*> \verbatim
*>          WORK is DOUBLE PRECISION array, dimension (max(1,2*N-2))
*>          If COMPZ = 'N', then WORK is not referenced.
*> \endverbatim
*>
*> \param[out] INFO
*> \verbatim
*>          INFO is INTEGER
*>          = 0:  successful exit
*>          < 0:  if INFO = -i, the i-th argument had an illegal value
*>          > 0:  the algorithm has failed to find all the eigenvalues in
*>                a total of 30*N iterations; if INFO = i, then i
*>                elements of E have not converged to zero; on exit, D
*>                and E contain the elements of a symmetric tridiagonal
*>                matrix which is orthogonally similar to the original
*>                matrix.
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \ingroup steqr
*
*  =====================================================================

      SUBROUTINE dsteqr( COMPZ, N, D, E, Z, LDZ, WORK, INFO )
*
*  -- LAPACK computational routine --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      CHARACTER          COMPZ
      INTEGER            INFO, LDZ, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   D( * ), E( * ), WORK( * ), Z( LDZ, * )
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE, TWO, THREE
      parameter( zero = 0.0d0, one = 1.0d0, two = 2.0d0,
     $                   three = 3.0d0 )
      INTEGER            MAXIT
      parameter( maxit = 30 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, ICOMPZ, II, ISCALE, J, JTOT, K, L, L1, LEND,
     $                   LENDM1, LENDP1, LENDSV, LM1, LSV, M, MM, MM1,
     $                   NM1, NMAXIT
      DOUBLE PRECISION   ANORM, B, C, EPS, EPS2, F, G, P, R, RT1, RT2,
     $                   S, SAFMAX, SAFMIN, SSFMAX, SSFMIN, TST
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      DOUBLE PRECISION   DLAMCH, DLANST, DLAPY2
      EXTERNAL           lsame, dlamch, dlanst, dlapy2
*     ..
*     .. External Subroutines ..
      EXTERNAL           dlae2, dlaev2, dlartg, dlascl, dlaset,
     $                   dlasr,
     $                   dlasrt, dswap, xerbla
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          abs, max, sign, sqrt
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      info = 0
*
      IF( lsame( compz, 'N' ) ) THEN
         icompz = 0
      ELSE IF( lsame( compz, 'V' ) ) THEN
         icompz = 1
      ELSE IF( lsame( compz, 'I' ) ) THEN
         icompz = 2
      ELSE
         icompz = -1
      END IF
      IF( icompz.LT.0 ) THEN
         info = -1
      ELSE IF( n.LT.0 ) THEN
         info = -2
      ELSE IF( ( ldz.LT.1 ) .OR. ( icompz.GT.0 .AND. ldz.LT.max( 1,
     $         n ) ) ) THEN
         info = -6
      END IF
      IF( info.NE.0 ) THEN
         CALL xerbla( 'DSTEQR', -info )
         RETURN
      END IF
*
*     Quick return if possible
*
      IF( n.EQ.0 )
     $   RETURN
*
      IF( n.EQ.1 ) THEN
         IF( icompz.EQ.2 )
     $      z( 1, 1 ) = one
         RETURN
      END IF
*
*     Determine the unit roundoff and over/underflow thresholds.
*
      eps = dlamch( 'E' )
      eps2 = eps**2
      safmin = dlamch( 'S' )
      safmax = one / safmin
      ssfmax = sqrt( safmax ) / three
      ssfmin = sqrt( safmin ) / eps2
*
*     Compute the eigenvalues and eigenvectors of the tridiagonal
*     matrix.
*
      IF( icompz.EQ.2 )
     $   CALL dlaset( 'Full', n, n, zero, one, z, ldz )
*
      nmaxit = n*maxit
      jtot = 0
*
*     Determine where the matrix splits and choose QL or QR iteration
*     for each block, according to whether top or bottom diagonal
*     element is smaller.
*
      l1 = 1
      nm1 = n - 1
*
   10 CONTINUE
      IF( l1.GT.n )
     $   GO TO 160
      IF( l1.GT.1 )
     $   e( l1-1 ) = zero
      IF( l1.LE.nm1 ) THEN
         DO 20 m = l1, nm1
            tst = abs( e( m ) )
            IF( tst.EQ.zero )
     $         GO TO 30
            IF( tst.LE.( sqrt( abs( d( m ) ) )*sqrt( abs( d( m+
     $          1 ) ) ) )*eps ) THEN
               e( m ) = zero
               GO TO 30
            END IF
   20    CONTINUE
      END IF
      m = n
*
   30 CONTINUE
      l = l1
      lsv = l
      lend = m
      lendsv = lend
      l1 = m + 1
      IF( lend.EQ.l )
     $   GO TO 10
*
*     Scale submatrix in rows and columns L to LEND
*
      anorm = dlanst( 'M', lend-l+1, d( l ), e( l ) )
      iscale = 0
      IF( anorm.EQ.zero )
     $   GO TO 10
      IF( anorm.GT.ssfmax ) THEN
         iscale = 1
         CALL dlascl( 'G', 0, 0, anorm, ssfmax, lend-l+1, 1, d( l ),
     $                n,
     $                info )
         CALL dlascl( 'G', 0, 0, anorm, ssfmax, lend-l, 1, e( l ), n,
     $                info )
      ELSE IF( anorm.LT.ssfmin ) THEN
         iscale = 2
         CALL dlascl( 'G', 0, 0, anorm, ssfmin, lend-l+1, 1, d( l ),
     $                n,
     $                info )
         CALL dlascl( 'G', 0, 0, anorm, ssfmin, lend-l, 1, e( l ), n,
     $                info )
      END IF
*
*     Choose between QL and QR iteration
*
      IF( abs( d( lend ) ).LT.abs( d( l ) ) ) THEN
         lend = lsv
         l = lendsv
      END IF
*
      IF( lend.GT.l ) THEN
*
*        QL Iteration
*
*        Look for small subdiagonal element.
*
   40    CONTINUE
         IF( l.NE.lend ) THEN
            lendm1 = lend - 1
            DO 50 m = l, lendm1
               tst = abs( e( m ) )**2
               IF( tst.LE.( eps2*abs( d( m ) ) )*abs( d( m+1 ) )+
     $             safmin )GO TO 60
   50       CONTINUE
         END IF
*
         m = lend
*
   60    CONTINUE
         IF( m.LT.lend )
     $      e( m ) = zero
         p = d( l )
         IF( m.EQ.l )
     $      GO TO 80
*
*        If remaining matrix is 2-by-2, use DLAE2 or SLAEV2
*        to compute its eigensystem.
*
         IF( m.EQ.l+1 ) THEN
            IF( icompz.GT.0 ) THEN
               CALL dlaev2( d( l ), e( l ), d( l+1 ), rt1, rt2, c,
     $                      s )
               work( l ) = c
               work( n-1+l ) = s
               CALL dlasr( 'R', 'V', 'B', n, 2, work( l ),
     $                     work( n-1+l ), z( 1, l ), ldz )
            ELSE
               CALL dlae2( d( l ), e( l ), d( l+1 ), rt1, rt2 )
            END IF
            d( l ) = rt1
            d( l+1 ) = rt2
            e( l ) = zero
            l = l + 2
            IF( l.LE.lend )
     $         GO TO 40
            GO TO 140
         END IF
*
         IF( jtot.EQ.nmaxit )
     $      GO TO 140
         jtot = jtot + 1
*
*        Form shift.
*
         g = ( d( l+1 )-p ) / ( two*e( l ) )
         r = dlapy2( g, one )
         g = d( m ) - p + ( e( l ) / ( g+sign( r, g ) ) )
*
         s = one
         c = one
         p = zero
*
*        Inner loop
*
         mm1 = m - 1
         DO 70 i = mm1, l, -1
            f = s*e( i )
            b = c*e( i )
            CALL dlartg( g, f, c, s, r )
            IF( i.NE.m-1 )
     $         e( i+1 ) = r
            g = d( i+1 ) - p
            r = ( d( i )-g )*s + two*c*b
            p = s*r
            d( i+1 ) = g + p
            g = c*r - b
*
*           If eigenvectors are desired, then save rotations.
*
            IF( icompz.GT.0 ) THEN
               work( i ) = c
               work( n-1+i ) = -s
            END IF
*
   70    CONTINUE
*
*        If eigenvectors are desired, then apply saved rotations.
*
         IF( icompz.GT.0 ) THEN
            mm = m - l + 1
            CALL dlasr( 'R', 'V', 'B', n, mm, work( l ),
     $                  work( n-1+l ),
     $                  z( 1, l ), ldz )
         END IF
*
         d( l ) = d( l ) - p
         e( l ) = g
         GO TO 40
*
*        Eigenvalue found.
*
   80    CONTINUE
         d( l ) = p
*
         l = l + 1
         IF( l.LE.lend )
     $      GO TO 40
         GO TO 140
*
      ELSE
*
*        QR Iteration
*
*        Look for small superdiagonal element.
*
*        Super important for my Julia implementation. We are happy to eat the
*        cost of looking for small superdiagonal elements
   90    CONTINUE
         IF( l.NE.lend ) THEN
            lendp1 = lend + 1
            DO 100 m = l, lendp1, -1
               tst = abs( e( m-1 ) )**2
               IF( tst.LE.( eps2*abs( d( m ) ) )*abs( d( m-1 ) )+
     $             safmin )GO TO 110
  100       CONTINUE
         END IF
*
         m = lend
*
  110    CONTINUE
         IF( m.GT.lend )
     $      e( m-1 ) = zero
         p = d( l )
         IF( m.EQ.l )
     $      GO TO 130
*
*        If remaining matrix is 2-by-2, use DLAE2 or SLAEV2
*        to compute its eigensystem.
*
         IF( m.EQ.l-1 ) THEN
            IF( icompz.GT.0 ) THEN
               CALL dlaev2( d( l-1 ), e( l-1 ), d( l ), rt1, rt2, c,
     $                      s )
               work( m ) = c
               work( n-1+m ) = s
               CALL dlasr( 'R', 'V', 'F', n, 2, work( m ),
     $                     work( n-1+m ), z( 1, l-1 ), ldz )
            ELSE
               CALL dlae2( d( l-1 ), e( l-1 ), d( l ), rt1, rt2 )
            END IF
            d( l-1 ) = rt1
            d( l ) = rt2
            e( l-1 ) = zero
            l = l - 2
            IF( l.GE.lend )
     $         GO TO 90
            GO TO 140
         END IF
*
         IF( jtot.EQ.nmaxit )
     $      GO TO 140
         jtot = jtot + 1
*
*        Form shift.
*
         g = ( d( l-1 )-p ) / ( two*e( l-1 ) )
         r = dlapy2( g, one )
         g = d( m ) - p + ( e( l-1 ) / ( g+sign( r, g ) ) )
*
         s = one
         c = one
         p = zero
*
*        Inner loop
*
         lm1 = l - 1
         DO 120 i = m, lm1
            f = s*e( i )
            b = c*e( i )
            CALL dlartg( g, f, c, s, r )
            IF( i.NE.m )
     $         e( i-1 ) = r
            g = d( i ) - p
            r = ( d( i+1 )-g )*s + two*c*b
            p = s*r
            d( i ) = g + p
            g = c*r - b
*
*           If eigenvectors are desired, then save rotations.
*
            IF( icompz.GT.0 ) THEN
               work( i ) = c
               work( n-1+i ) = s
            END IF
*
  120    CONTINUE
*
*        If eigenvectors are desired, then apply saved rotations.
*
         IF( icompz.GT.0 ) THEN
            mm = l - m + 1
            CALL dlasr( 'R', 'V', 'F', n, mm, work( m ),
     $                  work( n-1+m ),
     $                  z( 1, m ), ldz )
         END IF
*
         d( l ) = d( l ) - p
         e( lm1 ) = g
         GO TO 90
*
*        Eigenvalue found.
*
  130    CONTINUE
         d( l ) = p
*
         l = l - 1
         IF( l.GE.lend )
     $      GO TO 90
         GO TO 140
*
      END IF
*
*     Undo scaling if necessary
*
  140 CONTINUE
      IF( iscale.EQ.1 ) THEN
         CALL dlascl( 'G', 0, 0, ssfmax, anorm, lendsv-lsv+1, 1,
     $                d( lsv ), n, info )
         CALL dlascl( 'G', 0, 0, ssfmax, anorm, lendsv-lsv, 1,
     $                e( lsv ),
     $                n, info )
      ELSE IF( iscale.EQ.2 ) THEN
         CALL dlascl( 'G', 0, 0, ssfmin, anorm, lendsv-lsv+1, 1,
     $                d( lsv ), n, info )
         CALL dlascl( 'G', 0, 0, ssfmin, anorm, lendsv-lsv, 1,
     $                e( lsv ),
     $                n, info )
      END IF
*
*     Check for no convergence to an eigenvalue after a total
*     of N*MAXIT iterations.
*
      IF( jtot.LT.nmaxit )
     $   GO TO 10
      DO 150 i = 1, n - 1
         IF( e( i ).NE.zero )
     $      info = info + 1
  150 CONTINUE
      GO TO 190
*
*     Order eigenvalues and eigenvectors.
*
  160 CONTINUE
      IF( icompz.EQ.0 ) THEN
*
*        Use Quick Sort
*
         CALL dlasrt( 'I', n, d, info )
*
      ELSE
*
*        Use Selection Sort to minimize swaps of eigenvectors
*
         DO 180 ii = 2, n
            i = ii - 1
            k = i
            p = d( i )
            DO 170 j = ii, n
               IF( d( j ).LT.p ) THEN
                  k = j
                  p = d( j )
               END IF
  170       CONTINUE
            IF( k.NE.i ) THEN
               d( k ) = d( i )
               d( i ) = p
               CALL dswap( n, z( 1, i ), 1, z( 1, k ), 1 )
            END IF
  180    CONTINUE
      END IF
*
  190 CONTINUE
      RETURN
*
*     End of DSTEQR
*

      END
