# Copyright 2016 Singapore Management University (SMU). All Rights Reserved. 
#
# Permission to use, copy, modify and distribute this software and 
# its documentation for purposes of research, teaching and general
# academic pursuits, without fee and without a signed licensing
# agreement, is hereby granted, provided that the above copyright
# statement, this paragraph and the following paragraph on disclaimer
# appear in all copies, modifications, and distributions.  Contact
# Singapore Management University, Intellectual Property Management
# Office at iie@smu.edu.sg, for commercial licensing opportunities.
#
# This software is provided by the copyright holder and creator "as is"
# and any express or implied warranties, including, but not Limited to,
# the implied warranties of merchantability and fitness for a particular 
# purpose are disclaimed.  In no event shall SMU or the creator be 
# liable for any direct, indirect, incidental, special, exemplary or 
# consequential damages, however caused arising in any way out of the
# use of this software.


use warnings;
use strict;
use Data::Dumper;
use Math::Trig;
use Algorithm::LBFGS;
use Getopt::Long;

my $basis_function = 0;    #0: Gaussian, 1: Student-t
my $weighted = 1;          #0: non-weighted, 1: weighted
our $lambda = 10;
our $num_topics = 20;
our $data;
our $graph;
our $dim = 2;
our $num_iter = 100;
our $num_quasi_iter = 10;
our $output_file;
our $alpha;
our $beta;
our $gamma;

GetOptions ("num_topics=i" => \$num_topics,
			"dim=i" => \$dim,
            "lambda=f"   => \$lambda,
            "alpha=f"    => \$alpha,
            "beta=f"     => \$beta,
            "gamma=f"   => \$gamma,
            "basis_function=i"  => \$basis_function,
            "EM_iter=i" => \$num_iter,
            "Quasi_iter=i" => \$num_quasi_iter,
            "data=s" => \$data,
            "graph=s" => \$graph,
            "output_file=s" => \$output_file)
  or die("Error in command line arguments\n");

open(my $output, '>', $output_file) or die "Could not open file '$output_file' $!";

$alpha     = $alpha ? $alpha:0.01;
$gamma  = $gamma ? $gamma : 0.1 * $num_topics;

our $num_docs = 0;
our @input_lines;
my %Vocabulary;
open( F, "$data" ) or die "Couldn't load the data $data $!";
while (<F>) {
	chomp;
	my $line = $_;
	push(@input_lines, $line);
	$num_docs++;
}
close(F);
$beta = $beta ? $beta : 0.1 * $num_docs;

print $output "num_topics: ";
print $output $num_topics;
print $output "\n";

print $output "dim: ";
print $output $dim;
print $output "\n";

print $output "lambda: ";
print $output $lambda;
print $output "\n";

print $output "alpha: ";
print $output $alpha;
print $output "\n";

print $output "beta: ";
print $output $beta;
print $output "\n";

print $output "gamma: ";
print $output $gamma;
print $output "\n";

print $output "basis_function: ";
print $output $basis_function;
print $output "\n";

print $output "version (weighted or not): ";
print $output $weighted;
print $output "\n";

print $output "EM_iter: ";
print $output $num_iter;
print $output "\n";

print $output "Quasi_iter: ";
print $output $num_quasi_iter;
print $output "\n";

if($graph){
	print $output "graph: ";
	print $output $graph;
	print $output "\n";
}

my %all_tokens;
my %all_tokens_count;
my @doc_length;
my $c = 0;
foreach (@input_lines) {
	my $line = $_;
	my @tokens = split( /\s/, $line );
	$doc_length[$c] = @tokens;
	my @unique_tokens;
	my @count_unique_tokens;
	my $count = 0;
	my %hash_w;
	for ( my $i = 0 ; $i < @tokens ; $i++ ) {
		if(!defined $Vocabulary{$tokens[$i]}) {
			$Vocabulary{$tokens[$i]} = 1;
		}
		if(defined $hash_w{$tokens[$i]}) {
			$hash_w{$tokens[$i]} = $hash_w{$tokens[$i]} + 1;
		}
		else{
			$hash_w{$tokens[$i]} = 1;
		}
	}
	foreach my $name (keys %hash_w) {
   		$unique_tokens[$count] = $name;
   		$count_unique_tokens[$count] = $hash_w{$name};
   		$count++;
	}
	$all_tokens{$c} = \@unique_tokens;
	$all_tokens_count{$c} = \@count_unique_tokens;
	$c++;
}

our $wordsize = (keys %Vocabulary);
our %theta;
our %phi;
our %xai;
our %update_theta_numerator;
our @update_theta_denominator;

print $output "initialize phi \n";
for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
	my @position;
	for ( my $d = 0 ; $d < $dim ; $d++ ) {
		$position[$d]  = &gaussian_rand * 0.01;
		print $output $position[$d];
		print $output "\n";
	}
	print $output "---------------\n";
	$phi{$i}     = \@position;
}

print $output "initialize xai \n";
for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
	my @position;
	for ( my $d = 0 ; $d < $dim ; $d++ ) {
		$position[$d]  = &gaussian_rand * 0.01;
		print $output $position[$d];
		print $output "\n";
	}
	print $output "---------------\n";
	$xai{$i}     = \@position;
}

for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
	my @words;
	my $denominator = 0;
	for ( my $j = 0 ; $j < $wordsize ; $j++ ) {
		$words[$j] = -log( 1.0 - rand );
		$denominator += $words[$j];
	}
	for ( my $j = 0 ; $j < $wordsize ; $j++ ) {
		$words[$j] = $words[$j] / $denominator;
	}
	$theta{$i} = \@words;
}

our %pr_zgx;
our %pr_zgnm;
our %sum_zgnm_overw;
our %weights;

#load weights
if($graph){
	open( F,
	"$graph"
	) or die "Couldn't load the graph $graph $!";
	my $count_temp = 0;
	while (<F>) {
		chomp;
		my $line = $_;
		my @tokens = split( /\s/, $line );
		my @position;
		for ( my $d = 0 ; $d < $num_docs ; $d++ ) {
			$position[$d]  = $tokens[$d];
		}
		$weights{$count_temp} = \@position;
		$count_temp++;
	}
	close(F);
}

my $quasi = Algorithm::LBFGS->new;
$quasi->set_param(max_iterations => $num_quasi_iter);

my $eval_short = sub {
	my $x = shift;
	my $sum   = 0;
	my $docid = 0;
	my %grad;
	my @gradient;
	my @numerators;
	for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
		my @arr;
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			my $index = $j * $dim + $k;
			$arr[$k] = - $beta * $x->[$index];
			$gradient[$index] = 0;
		}
		$grad{$j} = \@arr;
	}
	for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
		my @prs;
		my $denominator = 0;
		my $offset_doc = $num_topics * $dim + $i*$dim;
		for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
			my $d = 0;
			my $offset_topic = $j * $dim;
			for ( my $k = 0 ; $k < $dim ; $k++ ) {
				$d += ($x->[$offset_doc + $k] - $x->[$offset_topic + $k])**2;
			}
			if($basis_function==0){
				$numerators[$j] = exp((-0.5) * $d );
			}
			else{
				$numerators[$j] =  1/((1 + $d)**1);
			}
			$denominator += $numerators[$j];
		}
		for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
			$prs[$j] = $numerators[$j] / $denominator;
		}
		$pr_zgx{$i} = \@prs;
		
		$docid = $i;
		my @tokens = values($all_tokens{$docid});
		my @count_tokens = values($all_tokens_count{$docid});
		 
		my $p_zgnm = $pr_zgnm{$docid};
		my $p_zgx  = $pr_zgx{$docid};
		for ( my $m = 0 ; $m < @tokens ; $m++ ) {
			my $p_znm = $p_zgnm->{$m};
			my $t = 0;
			for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
#				$t += 
#				  $p_znm->[$j] *
#				  log( $p_zgx->[$j]  *
#				  $theta{$j}->[ $tokens[$m] ]);
				$t += 
				  $p_znm->[$j] *
				  log( $p_zgx->[$j]);
			}
			$sum += $count_tokens[$m] * $t;
		}
		
		my $length = 0;
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$length += $x->[$offset_doc + $k]**2;
		}
		$sum += (-$gamma/2) * $length;
		
		my @gra;
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$gra[$k] = 0;
		}
		#gradient wrt phi
		for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
			my $offset_topic = $j * $dim;
			my $euclid = 0;
			for ( my $k = 0 ; $k < $dim ; $k++ ) {
				$euclid += ($x->[$offset_doc + $k] - $x->[$offset_topic + $k])**2;
			}			

			my $p_zgx = $pr_zgx{$docid}->[$j];
			for ( my $k = 0 ; $k < $dim ; $k++ ) {
				my $t;
				if($basis_function==0){
					$t = ( $doc_length[$docid] * $p_zgx - $sum_zgnm_overw{$docid}->[$j] ) * ( $x->[$offset_topic + $k] - $x->[$offset_doc + $k] );
				}
				else{
					$t = ( $doc_length[$docid] * $p_zgx - $sum_zgnm_overw{$docid}->[$j] ) * ( $x->[$offset_topic + $k] - $x->[$offset_doc + $k] ) * 2 * 1 /((1+$euclid)**1) ;
				}
				$grad{$j}->[$k] +=  $t;
				#wrt xai
				$gra[$k] += -$t;
			}
		}
		if($graph){
			my @grad_temp;
			for ( my $k = 0 ; $k < $dim ; $k++ ) {
				$grad_temp[$k] = 0;
			}
			my $offset = $num_topics * $dim;
			my $offset_doci = $offset + $i*$dim;
			for ( my $j = 0 ; $j < $num_docs ; $j++ ) {
				my $offset_docj = $offset + $j*$dim; 
				if($i!=$j){
					if($weights{$i}->[$j]!=0 || $weights{$j}->[$i]!=0){
						my $w = 1;
						if($weighted==1){
							$w = $weights{$i}->[$j]!=0 ? $weights{$i}->[$j]:$weights{$j}->[$i];
						}
						for ( my $k = 0 ; $k < $dim ; $k++ ) {
							$grad_temp[$k] += $w * ($x->[$offset_doci + $k] - $x->[$offset_docj + $k]);						
						}
					}
					else{
						my $length = 1;
						for ( my $k = 0 ; $k < $dim ; $k++ ) {
							$length += ($x->[$offset_doci + $k] - $x->[$offset_docj + $k])**2;
						}
						for ( my $k = 0 ; $k < $dim ; $k++ ) {
							$grad_temp[$k] += -($x->[$offset_doci + $k] - $x->[$offset_docj + $k])/($length**2); 
						}
					}
				}
			}
			for ( my $k = 0 ; $k < $dim ; $k++ ) {
				$grad_temp[$k] = -$lambda * $grad_temp[$k] * 2;
				$gra[$k] += $grad_temp[$k];
			}
		}
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$gradient[$offset_doc + $k] = -($gra[$k] - $gamma * $x->[$offset_doc + $k]);
		}
	}
	
	for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
		my $length = 0;
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$length += $x->[$j*$dim + $k]**2;
		}
		$sum += (-$beta/2) * $length;
	}
	
	#regularization
	if($graph){
		my $offset = $num_topics * $dim;
		for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
			my $offset_doci = $offset + $i*$dim;
			for ( my $j = $i ; $j < $num_docs ; $j++ ) {
				my $offset_docj = $offset + $j*$dim;
				if($i!=$j){
					my $euclid = 0;
					for ( my $k = 0 ; $k < $dim ; $k++ ) {
						$euclid += ($x->[$offset_doci + $k]-$x->[$offset_docj + $k])**2;
					}
					if($weights{$i}->[$j]!=0 || $weights{$j}->[$i]!=0){
						my $w = 1;
						if($weighted==1){
							$w = $weights{$i}->[$j]!=0 ? $weights{$i}->[$j]:$weights{$j}->[$i];
						}
						$sum -= $w * 2 * ($lambda/2) * $euclid;
					}
					else{
						$sum -= 2 * ($lambda/2)/($euclid + 1);
					}
				}
			}
		}
	}
	my $f = -$sum;
	print ($f);
	print ("\n");
	for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
		for ( my $k = 0 ; $k < $dim ; $k++ ) {
			$gradient[$j * $dim + $k] = -$grad{$j}->[$k];
		}
	}
	return ( $f, \@gradient );
};
#EM
for ( my $i = 0 ; $i < $num_iter ; $i++ ) {
	print($i);
	print("\n");
	&estep();
	&mstep();
}

# output to file
print $output "\n-------Xai-------\n";
for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
	for ( my $d = 0 ; $d < $dim ; $d++ ) {
		if($d==0){
			print $output $xai{$i}[$d];
		}
		else{
			print $output "\t";
			print $output $xai{$i}[$d];
		}
	}
	print $output "\n";
}

print $output "\n-------Phi-------\n";
for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
	for ( my $d = 0 ; $d < $dim ; $d++ ) {
		if($d==0){
			print $output $phi{$i}[$d];
		}
		else{
			print $output "\t";
			print $output $phi{$i}[$d];
		}
	}
}

print $output "\n-------Verbose-------\n";
my $docid = 0;
print $output "zgivend \n";
foreach (@input_lines) {
	my $p_zgx = $pr_zgx{$docid};
	for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
		print $output $p_zgx->[$j];
		print $output "\n";
	}
	print $output "-------------\n";
	$docid++;
}

print $output "Phi\n";
print $output Dumper(%phi);
print $output "Xai\n";
print $output Dumper(%xai);
print $output "Theta\n";
print $output Dumper(%theta);

sub update_theta {
	my $topic = shift;

	my $numerator = 0;
	for ( my $docid = 0 ; $docid < $num_docs ; $docid++ ) {
		my @tokens = values($all_tokens{$docid});
		my @count_tokens = values($all_tokens_count{$docid});
		
		my $p_zgnm = $pr_zgnm{$docid};
		for ( my $i = 0 ; $i < @tokens ; $i++ ) {
			my $p_znm = $p_zgnm->{$i};
			$update_theta_numerator{$topic}->[ $tokens[$i] ] += $count_tokens[$i] * $p_znm->[$topic];
		}
	}
};

sub batch_update_theta {
	my $topic = shift;
	&update_theta($topic );
	for ( my $j = 0 ; $j < $wordsize ; $j++ ) {
		$theta{$topic}->[$j] =
		  ($update_theta_numerator{$topic}->[$j] + $alpha ) /
		  ($update_theta_denominator[$topic] + $alpha * $wordsize);
	}
};

sub update_theta_denominator {
	my @result;

	for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
		my $denominator = 0;
		for ( my $docid = 0 ; $docid < $num_docs ; $docid++ ) {
			$denominator += $sum_zgnm_overw{$docid}->[$j];
		}
		push( @result, $denominator );
	}
	return @result;
};

sub mstep {
	@update_theta_denominator = update_theta_denominator();
	for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
		my @words;
		for ( my $j = 0 ; $j < $wordsize ; $j++ ) {
			$words[$j] = 0;
		}
		$update_theta_numerator{$i} = \@words;
	}
	for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
		&batch_update_theta($i);
	}
	#using quasi newton
	my @x0;
	for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
		for ( my $d = 0 ; $d < $dim ; $d++ ) {
			$x0[$i * $dim + $d] = $phi{$i}[$d];
		}
	}
	for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
		for ( my $d = 0 ; $d < $dim ; $d++ ) {
			$x0[$num_topics * $dim + $i * $dim + $d] = $xai{$i}[$d];
		}
	}
	print("start quasi\n");
	my $x;
	$x = $quasi->fmin($eval_short, \@x0);
	print("end quasi\n");
	for ( my $i = 0 ; $i < $num_topics ; $i++ ) {
		for ( my $d = 0 ; $d < $dim ; $d++ ) {
			$phi{$i}[$d] = $x->[$i * $dim + $d];
		}
	}
	for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
		for ( my $d = 0 ; $d < $dim ; $d++ ) {
			$xai{$i}[$d] = $x->[$num_topics * $dim + $i * $dim + $d];
		}
	}
	return;
};

sub estep {
	for ( my $i = 0 ; $i < $num_docs ; $i++ ) {
		my @prs;
		my $denominator = 0;
		my @numerators;
		for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
			my $euclid = 0;
			for ( my $k = 0 ; $k < $dim ; $k++ ) {
				$euclid += ($xai{$i}->[$k]-$phi{$j}->[$k])**2;
			}
			if($basis_function==0){
				$numerators[$j] = exp((-0.5) * $euclid);
			}
			else{
				$numerators[$j] = 1.0/((1 + $euclid)**1);
			}
			$denominator += $numerators[$j];
		}
		for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
			my $pr = $numerators[$j]/$denominator;
			$prs[$j] = $pr;
		}
		$pr_zgx{$i} = \@prs;
	}

	for ( my $docid = 0 ; $docid < $num_docs ; $docid++ ) {
		my @tokens = values($all_tokens{$docid});
		my @count_tokens = values($all_tokens_count{$docid});
		
		my %prs_znm;
		my @sum_zgnm;
		for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
			$sum_zgnm[$j] = 0;
		}
		my $p_zgx = $pr_zgx{$docid};
		for ( my $i = 0 ; $i < @tokens ; $i++ ) {
			my @prs;
			my $denominator = 0;
			my @numerator;
			for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
				my $t = $p_zgx->[$j] * $theta{$j}->[$tokens[$i]];
				$denominator += $t;
				$numerator[$j] = $t;
			}
			for ( my $j = 0 ; $j < $num_topics ; $j++ ) {
				my $p = $numerator[$j]/$denominator;
				$prs[$j] = $p;
				$sum_zgnm[$j] += $count_tokens[$i]*$p;
			}
			$prs_znm{$i} = \@prs;
		}
		$pr_zgnm{$docid} = \%prs_znm;
		$sum_zgnm_overw{$docid} = \@sum_zgnm;
	}
	return;
};


sub gaussian_rand {
	#This function is from Perl Cookbook, 2nd Edition (http://www.oreilly.com/catalog/perlckbk2/)
	
	my ( $u1, $u2 );    # uniformly distributed random numbers
	my $w;              # variance, then a weight
	my ( $g1, $g2 );    # gaussian-distributed numbers

	do {
		$u1 = 2 * rand() - 1;
		$u2 = 2 * rand() - 1;
		$w  = $u1 * $u1 + $u2 * $u2;
	} while ( $w >= 1 );

	$w  = sqrt( ( -2 * log($w) ) / $w );
	$g2 = $u1 * $w;
	$g1 = $u2 * $w;

	# return both if wanted, else just one
	return wantarray ? ( $g1, $g2 ) : $g1;
};
