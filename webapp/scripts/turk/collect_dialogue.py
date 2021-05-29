import sys
import boto3
import argparse
import pdb

import os

import sqlite3

from collections import defaultdict


def launch_hits(client, mturk_environment, num_hits, max_assignments, verbose=True):
    hit_ids = []
    hit_type_ids = []

    for i in range(num_hits):
        response = client.create_hit(
        MaxAssignments=max_assignments,
        LifetimeInSeconds= 3 * 24 * 60 * 60, # 3 days
        AssignmentDurationInSeconds= 20 * 60, # 20 minutes
        Reward=mturk_environment['reward'],
        Title='Find One In Common!',
        Keywords='game, chat, fun, AI, simple',
        Description='Your goal is to find a common object through natural language dialogue.',
        Question=question,
        QualificationRequirements=worker_requirements,
        #AutoApprovalDelayInSeconds:
        #HITReviewPolicy:
        )
        hit_id = response['HIT']['HITId']
        hit_ids.append(hit_id)
        hit_type_id = response['HIT']['HITTypeId']
        hit_type_ids.append(hit_type_id)

        if verbose:
            print "\nCreated HIT: {}".format(hit_id)
            print "\nYou can work the HIT here:"
            print mturk_environment['preview'] + "?groupId={}".format(hit_type_id)
            print("")

    return hit_ids, hit_type_ids


def print_hit(hit):
    for h_k in hit.keys():
        print("{}:{}".format(h_k, hit[h_k]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile_name', type=str, default=None)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--num_hits', type=int, default='1')
    parser.add_argument('--max_assignments', type=int, default='5')
    args = parser.parse_args()

    live = args.live
    environments = {
            "live": {
                "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
                "preview": "https://www.mturk.com/mturk/preview",
                "manage": "https://requester.mturk.com/mturk/manageHITs",
                "reward": "0.15"
            },
            "sandbox": {
                "endpoint": "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
                "preview": "https://workersandbox.mturk.com/mturk/preview",
                "manage": "https://requestersandbox.mturk.com/mturk/manageHITs",
                "reward": "0.15"
            },
    }
    if live:
        mturk_environment = environments["live"]
        db_path = "mturk.db"
    else:
        mturk_environment = environments["sandbox"]
        db_path = "mturk_sandbox.db"
    # use profile if one was passed as an arg, otherwise
    session = boto3.Session(profile_name=args.profile_name)
    client = session.client(
        service_name='mturk',
        region_name='us-east-1',
        endpoint_url=mturk_environment['endpoint'],
    )


    # Test that you can connect to the API by checking your account balance
    user_balance = client.get_account_balance()
    # In Sandbox this always returns $10,000. In live, it will be your acutal balance.
    print "Your account balance is {}".format(user_balance['AvailableBalance'])

    # The question we ask the workers is contained in this file.
    question = """
    <ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
    <ExternalURL>https://www.onecommon-tokyo.com/takuma/</ExternalURL>
    <FrameHeight>865</FrameHeight>
    </ExternalQuestion>
    """

    if live:
        worker_requirements = [{
            'QualificationTypeId': '000000000000000000L0',
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [90],
            'RequiredToPreview': True},
            {
            'QualificationTypeId': '00000000000000000071',
            'Comparator': 'EqualTo',
            'LocaleValues':[{
                'Country':"US"}]}
            ]
        total_dialogues = args.num_hits * args.max_assignments / 2
        total_cost = 1.2 * args.num_hits * args.max_assignments * float(mturk_environment["reward"])
        confirm = raw_input("launching hits in live! total dialogues: {} | cost: ${} | Continue? [Y|N]".format(total_dialogues, total_cost))
        assert args.max_assignments < 10, "hits_per_batch should be less than 10"
        assert total_cost < 200, "cost more than $200 at once"
        if confirm != 'Y':
            sys.exit()
    else:
        worker_requirements = [{
        'QualificationTypeId': '000000000000000000L0',
        'Comparator': 'GreaterThanOrEqualTo',
        'IntegerValues': [0],
        'RequiredToPreview': True}]

    hit_ids, hit_type_ids = launch_hits(client, mturk_environment, args.num_hits, args.max_assignments)